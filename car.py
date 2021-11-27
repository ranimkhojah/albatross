import websocket
import _thread
import time
import cv2
import numpy as np
# import websocket-client 

host = "albatross"
port = 8887

socket_address = f"ws://{host}:{port}/wsDrive"
video_address = f"http://{host}:{port}/video"


def on_message(ws, message):
    # print(message)
    pass


def on_error(ws, error):
    pass
    # print(error)


def on_close(ws, close_status_code, close_msg):
    # print("### closed ###")
    pass


def on_open(ws):
    def run(*args):
        # your car logic here

        cap = cv2.VideoCapture(video_address)

        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        while True:
            ret, frame = cap.read()
            # do something based on the frame
            low_b = np.uint8([5,5,5])
            high_b = np.uint8([0,0,0])
            mask = cv2.inRange(frame, high_b, low_b)
            # print(cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE))
            
            _, contours, hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)

            if len(contours) > 0 :
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] !=0 :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    print("CX : "+str(cx)+"  CY : "+str(cy))
                    if cx > 140 and cx <= 155:
                        print("Turn Right")
                        angle = 1.0
                        throttle = 0.2
                    elif cx > 130 and cx <= 140 :
                        print("On Track!")
                        angle = 0.0
                        throttle = 0.2
                    elif cx > 45 and cx <= 130 :
                        print("Turn Left")
                        angle = -1.0
                        throttle = 0.2
                    cv2.circle(frame, (cx,cy), 5, (255,255,255), -1)
            else:
                angle = 0.0
                throttle = 0.2

            message = f"{{\"angle\":{angle},\"throttle\":{throttle},\"drive_mode\":\"user\",\"recording\":false}}"
            ws.send(message)
            # print(message)

    _thread.start_new_thread(run, ())


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket_address,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()
