import argparse

from utils.datasets import *
from utils.utils import *
import threading
import time

people_count = 0

from datetime import datetime
import schedule
import numpy as np


# Get the current date and time
now = datetime.now()

# Format the date and time using the strftime function
file_name = now.strftime("%d_%B_%Y")



def calculate_day_crowd_sum():

    print('EOD Function Triggered Closing all files')
    sum = 0 
    crowd_count = []
    crowd_log = []
    file_data= open("logs/CountLog"+file_name+".txt",'r').read()
    for line in file_data.split('\n'):
        try:
            sum = sum + int(line.split(" ")[2])
            crowd_count.append(int(line.split(" ")[2]))
            crowd_log.append(line)
        except Exception as e:
            pass
            
    sum_file = open("logs/SumOfPeopleCountLog"+file_name+".txt",'w')
    sum_file.write(str(sum))
    sum_file.close()
    print("\n\nSumOfPeopleCountLog"+file_name+":  "+str(sum))

    ind = np.argmax(crowd_count)
    max_file = open("logs/MaxPeopleCountLog"+file_name+".txt",'w')
    max_file.write(crowd_log[ind])
    max_file.close()
    print("\n\n"+"MaxPeopleCountLog"+file_name+":  "+str(crowd_count[ind]))
    print('Completed..!!')

    os._exit(0)


def update_value():
    global people_count
    log_file = open("logs/CountLog"+file_name+".txt",'w')
    log_file.close()

    while True:
        # Get the current date and time
        now = datetime.now()    
        log_date = now.strftime("%d_%B_%Y %I:%M:%S.%f%p")
        print(log_date+' '+str(people_count))
        log_file = open("logs/CountLog"+file_name+".txt",'a+')
        log_file.write(log_date+' '+str(people_count)+"\n")
        log_file.close()
 
        time.sleep(5)



schedule.every().day.at(str(input('\n\n\nEnter EOD time to terminate the program (EX: 10:00,  23:59. 02:05): '))).do(calculate_day_crowd_sum)

def check_end_of_day():
    while True:
        schedule.run_pending()
        time.sleep(1)


def detect(save_img=False):

    global intermediate,high,normal,people_count
    
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
    
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # List to store bounding coordinates of people
        people_coords = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        if label is not None:
                            if (label.split())[0] == 'person':
                                people_coords.append(xyxy)
                                plot_one_box(xyxy, im0, line_thickness=3)
                                
            # cv2.rectangle(im0s, (0,0),(1920,150), (0,0,0), -1, cv2.LINE_AA)  # filled
            people_count = int(len(people_coords))
            
            try:
                cv2.putText(im0s, "Total people : {:2d}".format(int(len(people_coords))), (75,75), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 5, cv2.LINE_AA) 
            except:
                cv2.putText(im0, "Total people : {:2d}".format(int(len(people_coords))), (75,75), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 5, cv2.LINE_AA) 
                pass
            cv2.namedWindow(p, cv2.WINDOW_NORMAL)
            cv2.imshow(p, im0)
            if opt.img:
                key = cv2.waitKey(0)
                if key==ord('q'):
                    os._exit(0)
                    raise StopIteration
                else:
                    cv2.destroyAllWindows()
                    continue
            else:
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    os._exit(0)


    # print('Done. (%.3fs)' % (time.time() - t0))
    # os._exit(0)

if __name__ == '__main__':


    source_file = input('\nEnter input source type 0 for the webcam or enter the video file path for the video: ').replace('"','')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default=source_file, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--img', type=bool, default=False, help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    
    thread = threading.Thread(target=update_value)
    thread.start()

    thread1 = threading.Thread(target=check_end_of_day)
    thread1.start()

    with torch.no_grad():
        detect()
