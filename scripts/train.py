from __future__ import division

import argparse
import os

import tqdm
import torch
from terminaltables import AsciiTable
from torch.autograd import Variable

from torch_yolo3.models import Darknet
from torch_yolo3.datasets import ListDataset
from torch_yolo3.logger import Logger
from torch_yolo3.parse_config import parse_data_config
from torch_yolo3.utils import load_classes, weights_init_normal, update_path
from scripts.eval import evaluate

METRICS = [
    "grid_size", "loss",
    "x", "y", "w", "h",
    "cls", "cls_acc",
    "recall50", "recall75", "precision",
    "conf", "conf_obj", "conf_noobj",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_config, model_def, trained_weights, multiscale_training,
         img_size, grad_accums, evaluation_interval, checkpoint_interval,
         batch_size, epochs, path_output, nb_cpu):
    logger = Logger("logs")

    os.makedirs(path_output, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(update_path(data_config))
    train_path = update_path(data_config["train"])
    valid_path = update_path(data_config["valid"])
    class_names = load_classes(update_path(data_config["names"]))

    # Initiate model
    model = Darknet(update_path(model_def)).to(DEVICE)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if trained_weights:
        if trained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(trained_weights))
        else:
            model.load_darknet_weights(trained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=multiscale_training, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nb_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        # start_time = time.time()
        pbar = tqdm.tqdm(total=len(dataloader))
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            model, loss = training_batch(dataloader, model, optimizer, epochs,
                                         epoch, batch_i, imgs, targets, grad_accums, logger)
            pbar.set_description("training batch loss=%.5f" % loss.item())
            pbar.update()

        if epoch % evaluation_interval == 0:
            evaluate_epoch(model, valid_path, img_size, batch_size, epoch, class_names, logger)

        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(path_output, "yolov3_ckpt_%05d.pth" % epoch))


def training_batch(dataloader, model, optimizer, epochs, epoch, batch_i, imgs, targets, grad_accums, logger, verbose=0):
    batches_done = len(dataloader) * epoch + batch_i

    imgs = Variable(imgs.to(DEVICE))
    targets = Variable(targets.to(DEVICE), requires_grad=False)

    loss, outputs = model(imgs, targets)
    loss.backward()

    if batches_done % grad_accums == 0:
        # Accumulates gradient before each step
        optimizer.step()
        optimizer.zero_grad()

    if verbose:
        # Log progress
        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(METRICS):
            metric_table, tensorboard_log = metrics_export(metric_table, model, loss, metric)
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        # epoch_batches_left = len(dataloader) - (batch_i + 1)
        # time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        # log_str += f"\n---- ETA {time_left}"
        print(log_str)

    model.seen += imgs.size(0)
    return model, loss


def evaluate_epoch(model, valid_path, img_size, batch_size, epoch, class_names, logger):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=img_size,
        batch_size=batch_size,
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")


def metrics_export(metric_table, model, loss, metric):
    formats = {m: "%.6f" for m in METRICS}
    formats["grid_size"] = "%2d"
    formats["cls_acc"] = "%.2f%%"
    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
    metric_table += [[metric, *row_metrics]]

    # Tensorboard logging
    tensorboard_log = []
    for j, yolo in enumerate(model.yolo_layers):
        for name, metric in yolo.metrics.items():
            if name != "grid_size":
                tensorboard_log += [(f"{name}_{j + 1}", metric)]
    tensorboard_log += [("loss", loss.item())]
    return metric_table, tensorboard_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--grad_accums", type=int, default=2, help="number of gradient accumulations before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--path_output", type=str, default="output", help="path to output folder")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path_img to data config file")
    parser.add_argument("--trained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--nb_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    main(data_config=opt.data_config, model_def=opt.model_def, trained_weights=opt.trained_weights,
         multiscale_training=opt.multiscale_training, img_size=opt.img_size, grad_accums=opt.grad_accums,
         evaluation_interval=opt.evaluation_interval, checkpoint_interval=opt.checkpoint_interval,
         batch_size=opt.batch_size, epochs=opt.epochs, path_output=opt.path_output, nb_cpu=opt.nb_cpu)
