import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from realtime.model_def import ByteSessionClassifier
from session_data import BYTE_PAD_TOKEN, ByteSessionDataset, build_session_manifest


def iter_with_progress(iterable, total, desc: str, enable: bool):
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="Train ByteSessionClassifier")
    parser.add_argument("--dataset-root", default=os.path.join("dataset", "iscx"))
    parser.add_argument(
        "--session-root", default=os.path.join("processed_full", "sessions")
    )
    parser.add_argument(
        "--manifest", default=os.path.join("processed_full", "sessions", "manifest.csv")
    )
    parser.add_argument("--rebuild-sessions", action="store_true")
    parser.add_argument("--num-packets", type=int, default=12)
    parser.add_argument("--packet-len", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--flow-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-flows-per-class", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=8)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="alias of --max_epoch (kept for backward compatibility)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early stopping patience on test_acc (<=0 disables)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="inverse-frequency weight power, 0 disables class weighting",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    parser.add_argument(
        "--model-out",
        default=os.path.join("pytorch_model", "byte_session_classifier.pth"),
    )
    parser.add_argument(
        "--log-out",
        default=os.path.join("pytorch_model", "byte_session_train_log.csv"),
    )
    parser.add_argument(
        "--report-out",
        default=os.path.join("pytorch_model", "byte_session_train_report.json"),
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str):
    if device_arg:
        wanted = str(device_arg).strip().lower()
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable, fallback to CPU")
            return torch.device("cpu")
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_class_weights(rows, num_classes: int, power: float) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for row in rows:
        label = int(row["label"])
        if 0 <= label < num_classes:
            counts[label] += 1.0

    if power <= 0:
        return torch.ones((num_classes,), dtype=torch.float32)

    counts = np.clip(counts, a_min=1.0, a_max=None)
    inv = np.power(1.0 / counts, power)
    norm = inv * (num_classes / np.sum(inv))
    return torch.tensor(norm, dtype=torch.float32)


def compute_per_class_metrics(conf_mat: np.ndarray):
    metrics = []
    precisions = []
    recalls = []
    f1s = []
    num_classes = conf_mat.shape[0]

    for i in range(num_classes):
        tp = float(conf_mat[i, i])
        fp = float(conf_mat[:, i].sum() - tp)
        fn = float(conf_mat[i, :].sum() - tp)
        support = int(conf_mat[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        metrics.append(
            {
                "class_id": i,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }
    return metrics, macro


def evaluate(
    model, loader, device, criterion, num_classes: int, show_progress: bool = True
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        loop = iter_with_progress(
            loader,
            total=len(loader),
            desc="eval",
            enable=show_progress,
        )
        for batch in loop:
            x = batch["bytes"].to(device, non_blocking=True)
            mask = batch["packet_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits, _ = model(x, mask)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total_correct += pred.eq(y).sum().item()
            total += y.size(0)

            y_np = y.detach().cpu().numpy().astype(np.int64)
            p_np = pred.detach().cpu().numpy().astype(np.int64)
            for yi, pi in zip(y_np, p_np):
                if 0 <= yi < num_classes and 0 <= pi < num_classes:
                    conf_mat[yi, pi] += 1

    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = total_correct / total if total > 0 else 0.0
    per_class, macro = compute_per_class_metrics(conf_mat)
    return avg_loss, avg_acc, per_class, macro


def main():
    args = parse_args()
    if args.epochs is not None:
        args.max_epoch = int(args.epochs)
    set_seed(args.seed)
    show_progress = not args.no_progress

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)

    if args.rebuild_sessions or not os.path.exists(args.manifest):
        args.manifest = build_session_manifest(
            dataset_root=args.dataset_root,
            output_root=args.session_root,
            num_packets=args.num_packets,
            packet_len=args.packet_len,
            pad_token=BYTE_PAD_TOKEN,
            train_ratio=args.train_ratio,
            flow_timeout_s=args.flow_timeout_s,
            max_flows_per_class=args.max_flows_per_class,
            seed=args.seed,
        )

    train_ds = ByteSessionDataset(args.manifest, split="train")
    test_ds = ByteSessionDataset(args.manifest, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Class count derives from manifest labels to avoid hard-coded coupling.
    max_label = max(int(r["label"]) for r in train_ds.rows + test_ds.rows)
    num_classes = max_label + 1

    device = get_device(args.device)
    model = ByteSessionClassifier(num_classes=num_classes).to(device)

    class_weights = build_class_weights(
        train_ds.rows,
        num_classes=num_classes,
        power=args.class_weight_power,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    with open(args.log_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "epoch_seconds",
            ]
        )

    best_test_acc = -1.0
    epochs_without_improvement = 0
    best_epoch = 0
    best_macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_per_class = []
    epoch_loop = iter_with_progress(
        range(args.max_epoch),
        total=args.max_epoch,
        desc="epochs",
        enable=show_progress,
    )

    for epoch in epoch_loop:
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        train_loop = iter_with_progress(
            train_loader,
            total=len(train_loader),
            desc=f"train epoch {epoch + 1}/{args.max_epoch}",
            enable=show_progress,
        )
        for batch in train_loop:
            x = batch["bytes"].to(device, non_blocking=True)
            mask = batch["packet_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits, _ = model(x, mask)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += pred.eq(y).sum().item()
            total += y.size(0)

            if show_progress:
                set_postfix = getattr(train_loop, "set_postfix", None)
                if callable(set_postfix):
                    set_postfix(
                        loss=f"{(total_loss / max(total, 1)):.4f}",
                        acc=f"{(100.0 * total_correct / max(total, 1)):.2f}%",
                    )

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = total_correct / total if total > 0 else 0.0

        test_loss, test_acc, per_class, macro = evaluate(
            model,
            test_loader,
            device,
            criterion=criterion,
            num_classes=num_classes,
            show_progress=show_progress,
        )
        scheduler.step(test_loss)
        lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - t0

        with open(args.log_out, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    f"{lr:.8f}",
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{test_loss:.6f}",
                    f"{test_acc:.6f}",
                    f"{epoch_seconds:.2f}",
                ]
            )

        print(
            "[Epoch {}/{}] train_loss={:.6f} train_acc={:.2f}% test_loss={:.6f} test_acc={:.2f}% time={:.2f}s".format(
                epoch + 1,
                args.max_epoch,
                train_loss,
                train_acc * 100.0,
                test_loss,
                test_acc * 100.0,
                epoch_seconds,
            )
        )
        print(
            "           macro_precision={:.4f} macro_recall={:.4f} macro_f1={:.4f} lr={:.2e}".format(
                macro["precision"],
                macro["recall"],
                macro["f1"],
                lr,
            )
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_macro = dict(macro)
            best_per_class = [dict(x) for x in per_class]
            torch.save(model.state_dict(), args.model_out)
        else:
            epochs_without_improvement += 1

        if show_progress:
            set_postfix = getattr(epoch_loop, "set_postfix", None)
            if callable(set_postfix):
                set_postfix(
                    best_acc=f"{(100.0 * best_test_acc):.2f}%",
                    wait=f"{epochs_without_improvement}/{max(args.patience, 0)}",
                )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"[EARLY STOP] no improvement in {args.patience} epochs, "
                f"best_test_acc={best_test_acc:.4f}"
            )
            break

    report = {
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "num_classes": num_classes,
        "class_weight_power": args.class_weight_power,
        "class_weights": [float(w) for w in class_weights.detach().cpu().numpy()],
        "macro": best_macro,
        "per_class": best_per_class,
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[DONE] best_test_acc={best_test_acc:.4f} best_epoch={best_epoch} "
        f"model={args.model_out} report={args.report_out}"
    )


if __name__ == "__main__":
    main()
