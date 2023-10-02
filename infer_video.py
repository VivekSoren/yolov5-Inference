import argparse
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / 'yolov5s.onnx',
    source=ROOT / 'data/video', 
    data=ROOT / 'data/coco128.yaml',
    imgsz=(640, 640), 
    conf_thres=0.25,
    device='', 
    view_img=False, 
    save_txt=False,
    save_crop=False, 
    nosave=False, 
    visualize=False, 
    project=ROOT / 'runs/inference', 
    name='exp', 
    exist_ok=False, 
    line_thickness=3, 
    hide_labels=False, 
    hide_conf=False, 
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, onnx = model.stride, model.names, model.onnx
    imgsz = check_img_size(imgsz, s=stride)
    
    # Dataloader
    bs = 1  # Batch Size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=onnx, vid_stride=1)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run Inference
    model.warmup(imgsz=(1 if onnx or model.triton else bs, 3, *imgsz))
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]: 
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=False)
            pred = model(im, augment=False, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            p, im0 = path, im0s.copy()

            p = Path(p)
            save_path = str(save_dir / p.name)
            s += '%gx%g ' % im.shape[2:]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f'{names[c]}'
                
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                  
            im0 = annotator.result()
            if save_img:
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.onnx', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/video', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    