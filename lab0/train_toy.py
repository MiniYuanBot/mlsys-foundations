"""L00: supplied data/model/CLI/checkpoint; complete train_step and evaluate."""
import argparse
import torch
from torch import nn
from common.io import write_csv
from common.records import run_directory


def make_data():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2048, 16, generator=generator)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x[:1600], y[:1600], x[1600:], y[1600:]


def make_model():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))


def train_step(model, optimizer, x, y):
    """Return (mean_loss: float, correct_count: int), measured before the update."""
    # TODO L00-1: train(), zero_grad(), logits, cross entropy, backward(), step().
    # Detach scalars; don't retain computation graphs in your history.
    raise NotImplementedError("L00-1: implement one training step")


def evaluate(model, x, y, batch_size=64):
    """Return loss (sample-weighted mean), accuracy, num_samples; use eval/no_grad."""
    # TODO L00-2: account for a short last batch; visit every input exactly once.
    raise NotImplementedError("L00-2: implement evaluation")


def save_checkpoint(path, model, config):
    torch.save({"model": model.state_dict(), "config": config, "seed": config["seed"]}, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(payload["model"])
    return model, payload["config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory, e.g. results/lab0/run1")
    args = parser.parse_args()
    config = dict(seed=42, epochs=20, batch_size=64, lr=0.01, device="cpu",
                  dtype="float32", model_revision="not applicable", tokenizer_revision="not applicable")
    torch.manual_seed(config["seed"])
    train_x, train_y, test_x, test_y = make_data()
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    generator = torch.Generator().manual_seed(config["seed"])
    rows = []
    with run_directory(args.out, config) as out:
        for epoch in range(1, config["epochs"] + 1):
            permutation = torch.randperm(len(train_x), generator=generator)
            loss_sum, correct = 0.0, 0
            for ids in permutation.split(config["batch_size"]):
                loss, count = train_step(model, optimizer, train_x[ids], train_y[ids])
                loss_sum += loss * len(ids)
                correct += count
            metrics = evaluate(model, test_x, test_y)
            row = dict(epoch=epoch, train_loss=loss_sum / len(train_x),
                       train_accuracy=correct / len(train_x), test_accuracy=metrics["accuracy"])
            rows.append(row)
            print(row)
        write_csv(out / "epochs.csv", rows)
        write_csv(out / "raw.csv", rows)
        write_csv(out / "summary.csv", [rows[-1]])
        save_checkpoint(out / "model.pt", model, config)
        restored, _ = load_checkpoint(out / "model.pt")
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(test_x[:32]), restored(test_x[:32]), rtol=1e-6, atol=1e-6)
        print("checkpoint roundtrip passed")


if __name__ == "__main__":
    main()

