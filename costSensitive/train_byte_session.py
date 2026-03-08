import argparse
import csv
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
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
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device, show_progress: bool = True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

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

    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = total_correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def main():
    args = parse_args()
    set_seed(args.seed)
    show_progress = not args.no_progress

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_out), exist_ok=True)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_correct = 0
        total = 0

        train_loop = iter_with_progress(
            train_loader,
            total=len(train_loader),
            desc=f"train epoch {epoch + 1}/{args.epochs}",
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

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            device,
            show_progress=show_progress,
        )
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
                args.epochs,
                train_loss,
                train_acc * 100.0,
                test_loss,
                test_acc * 100.0,
                epoch_seconds,
            )
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), args.model_out)

    print(f"[DONE] best_test_acc={best_test_acc:.4f} model={args.model_out}")


if __name__ == "__main__":
    main()
