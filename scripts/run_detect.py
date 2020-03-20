"""Run detection on custom image folder.

Example
-------
python3 run_detect.py \
    --image_folder ./data_images \
    --model_def ./config/yolov3.cfg \
    --weights_path ./weights/yolov3.weights \
    --output_folder ./output \
    --class_path ./data/coco.names \
    --batch_size 5 \
    --nb_cpu 9

"""

from __future__ import division

import argparse
import logging
import os
from functools import partial

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.ticker import NullLocator
from pathos.multiprocessing import ProcessPool
from torch.autograd import Variable
from torch.utils.data import DataLoader

# sys.path += [os.path.abspath('..'), os.path.abspath('.')]
from torch_yolo3.models import Darknet
from torch_yolo3.datasets import ImageFolder
from torch_yolo3.utils import NB_CPUS, load_classes, rescale_boxes
from torch_yolo3.evaluate import non_max_suppression


def main(image_folder, model_def, weights_path, class_path, output_folder, img_size,
         conf_thres, nms_thres, batch_size, nb_cpu):
    # use GPU if it is possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # at least one cpu has to be set
    nb_cpu = max(1, nb_cpu)
    # prepare the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    img_folder = ImageFolder(image_folder, img_size=img_size)
    dataloader = DataLoader(img_folder, batch_size=batch_size, shuffle=False, num_workers=nb_cpu)

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_paths = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    pbar = tqdm.tqdm(total=len(img_folder), desc='Performing object detection')
    for path_imgs, input_imgs in dataloader:
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detects = model(input_imgs)
            detects = non_max_suppression(detects, conf_thres, nms_thres)

        # Save image and detections
        img_paths.extend(path_imgs)
        img_detections.extend(detects)
        pbar.update(len(path_imgs))
    pbar.close()

    # Bounding-box colors
    cmap = plt.get_cmap("jet")
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
    # np.random.shuffle(colors)

    _wrap_export = partial(wrap_export_detection, img_size=img_size, colors=colors,
                           classes=classes, output_folder=output_folder)
    with ProcessPool(nb_cpu) as pool:
        # Iterate through images and save plot of detections
        list(tqdm.tqdm(pool.imap(_wrap_export, zip(img_paths, img_detections)),
                       desc='Saving images'))


def wrap_export_detection(img_detections, img_size, colors, classes, output_folder):
    path_img, detections = img_detections
    return export_detections(path_img, detections, img_size, colors, classes, output_folder)


def export_detections(path_img, detections, img_size, colors, classes, output_folder):
    # Create figure
    img = plt.imread(path_img)
    img_height, img_width = img.shape[:2]
    fig = plt.figure()
    fig.gca().imshow(img)

    raw_detect = []
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            box_width = float(x2 - x1)
            box_height = float(y2 - y1)

            color = colors[int(cls_pred)]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_width, box_height, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            fig.gca().add_patch(bbox)
            # Add label
            text_fmt = dict(color="white", fontsize=8, verticalalignment="top",
                            bbox={"color": color, "pad": 0, "alpha": 0.5})
            fig.gca().text(x1, y1, s=classes[int(cls_pred)], **text_fmt)
            fig.gca().text(x2 - 40, y1, s=str(np.round(conf.numpy(), 2)), **text_fmt)

            box_centre_x = float((x2 + x1) / 2)
            box_centre_y = float((y2 + y1) / 2)
            bbox = [int(cls_pred),
                    np.round(box_centre_x / img_width, 5), np.round(box_centre_y / img_height, 5),
                    np.round(box_width / img_width, 5), np.round(box_height / img_height, 5)]
            raw_detect.append(bbox)

    # export detection in the COCO format (the same as training)
    img_name, _ = os.path.splitext(os.path.basename(path_img))
    with open(os.path.join(output_folder, img_name + '.txt'), 'w') as fp:
        fp.write(os.linesep.join([' '.join(map(str, det)) for det in raw_detect]))

    # Save generated image with detections
    fig.gca().axis('off')
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())
    filename, _ = os.path.splitext(os.path.basename(path_img))
    fig.savefig(os.path.join(output_folder, f"{filename}.jpg"), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path_img to dataset")
    parser.add_argument("--output_folder", type=str, default="output", help="path_img to output")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path_img to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path_img to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path_img to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--nb_cpu", type=int, default=NB_CPUS,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path_img to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    main(image_folder=opt.image_folder, model_def=opt.model_def, weights_path=opt.weights_path,
         class_path=opt.class_path, output_folder=opt.output_folder, img_size=opt.img_size,
         conf_thres=opt.conf_thres, nms_thres=opt.nms_thres,
         batch_size=opt.batch_size, nb_cpu=opt.nb_cpu)
    print("Done :]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cli()
