import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def xywh_ltwh( xywh ):
    x, y, w, h = xywh
    l = x - w/2
    u = y - h/2
    return int(l), int(u), int(w), int(h)


def to_bbs( result ):
    boxes = result.boxes
    _bbs = list(   zip( map( xywh_ltwh, boxes.xywh.cpu().numpy() )
                 , boxes.conf.cpu().numpy()
                 , boxes.cls.cpu().numpy().astype('int')
              ) )

    return _bbs


def draw_bbs( bbs, frame ):
    for ele in bbs:
        x, y, w, h = map( int, ele[0] )
        frame = cv2.rectangle(frame, (x, y,), (x + w, y + h,), (255, 0, 0,), 2)

        text = 'cls: {}, con: {}.'.format( str(ele[2]), ele[1] )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        font_thickness = 2

        # Specify the text and position
        position = (x + 10, y + 10)
        # Use putText to write text on the image
        frame = cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        
        return frame
    

def draw_track( track, frame ):

    x, y, w, h = map( int, track.to_ltwh() )
    frame = cv2.rectangle(frame, (x, y,), (x + w, y + h,), (255, 0, 0,), 6)

    if track.original_ltwh is not None:
        x, y, w, h = map( int, track.original_ltwh )
        frame = cv2.rectangle(frame, (x, y,), (x + w, y + h,), (0, 0, 255,), 2)

    text = 'id: {}, ag: {}.'.format( str(track_id), track.age )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    font_thickness = 2

    # Specify the text and position
    position = (x + 10, y + 10)
    # Use putText to write text on the image
    frame = cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)


if __name__ == '__main__':
    cap = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.100.208:554/Streaming/Channels/101?transportmode=unicast")

    scale_factor = 0.7

    model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

    tracker = DeepSort(
        max_iou_distance=0.5,
        max_age=60,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        gating_only_position=False,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None,
    )


    while True:
        # Read a frame from the video source
        ret, frame = cap.read()
        

        # Check if the frame is successfully read
        if not ret:
            print("Error: Could not read frame.")
            break


        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)


        bbs = to_bbs( model( frame, classes=[0,1,2], conf=0.3 )[0] )  # list of Results objects
        tracks = tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            draw_track( track, frame )

        # draw_bbs( bbs, frame )
        

        # Display the frame
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
