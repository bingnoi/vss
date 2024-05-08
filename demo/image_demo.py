from argparse import ArgumentParser

import sys
sys.path.append("/datadisk/lixinhao/vss")

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    # parser.add_argument('label', help='label file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    import os

    from pathlib import Path
    img = Path(args.img)

    for root, dirs, files in os.walk(img):
        for file in files:
            if "val" in root:
                value = str(file)
                # print(root,dirs,value)
                root_dir = root.split("/")[-2]
                img_value = f"{root}/{value}"
                result = inference_segmentor(model, img_value)

                # label = value.split(".")[0]
                root_name = root.replace("leftImg8bit","gtFine")
                label_name = file.replace("_leftImg8bit.png","_gtFine_labelTrainIds.png")
                label = f"{root_name}/{label_name}"

                # show the results
                show_result_pyplot(value,model, img_value, result,label, get_palette(args.palette))

    # for root, dirs, files in os.walk(img):
    #     for file in files:
    #         if file.endswith(".jpg"):
    #             value = str(file)
    #             # print(root,dirs,value)
    #             root_dir = root.split("/")[-2]
    #             img_value = f"{root}/{value}"
    #             result = inference_segmentor(model, img_value)

    #             # label = value.split(".")[0]
    #             root_name = root.replace("origin","mask")
    #             label_name = file.replace(".jpg",".png")
    #             label = f"{root_name}/{label_name}"
    #             # show the results
    #             show_result_pyplot(value,model, img_value, result,label, get_palette(args.palette))


if __name__ == '__main__':
    main()
