import cv2
import dlib

'''My first AI program from learning Youtube'''

# read img
img = cv2.imread("anhVu.png")

# convert img to 2D
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# dlib: face recognition detector
face_detector = dlib.get_frontal_face_detector()

# load predictor
face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# use detector to find face landmarks
faces = face_detector(gray)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # draw rectangle
    cv2.rectangle(img=img, pt1=(x1,y1), pt2=(x2, y2), color=(255,0,0), thickness=3)

    # loop through 68 points and draw circle
    face_features = face_predictor(image=gray, box=face)
    for n in range(0, 68):
        x = face_features.part(n).x
        y = face_features.part(n).y
        cv2.circle(img=img, center=(x, y), radius=2, color=(0,255,0), thickness=2)

cv2.imshow(winname="Face Recognition Program", mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
