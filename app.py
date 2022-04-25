from flask import Flask, render_template, Response
import cv2
import dlib
from math import hypot

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frames():
    nose_image = cv2.imread('dlib/pig_nose.png')

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # loading face detector
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(
                'dlib/shape_predictor_68_face_landmarks.dat')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(frame)
            for face in faces:
                landmarks = predictor(gray_frame, face)

                # Nose coordinates
                top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                right_nose = (landmarks.part(35).x, landmarks.part(35).y)

                nose_width = int(hypot(left_nose[0] - right_nose[0],
                                       left_nose[1] - right_nose[1]) * 1.7)
                nose_height = int(nose_width * 0.77)

                # New nose position
                top_left = (int(center_nose[0] - nose_width / 2),
                            int(center_nose[1] - nose_height / 2))
                bottom_right = (int(center_nose[0] + nose_width / 2),
                                int(center_nose[1] + nose_height / 2))

                # Adding the new nose
                nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
                nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                _, nose_mask = cv2.threshold(
                    nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

                nose_area = frame[top_left[1]: top_left[1] + nose_height,
                                  top_left[0]: top_left[0] + nose_width]
                nose_area_no_nose = cv2.bitwise_and(
                    nose_area, nose_area, mask=nose_mask)
                final_nose = cv2.add(nose_area_no_nose, nose_pig)

                frame[top_left[1]: top_left[1] + nose_height,
                      top_left[0]: top_left[0] + nose_width] = final_nose

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
