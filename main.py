import cv2
import sys

imagePath = sys.argv[1]
cascPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

if len(faces) == 0:
    print("There is no faces")
    exit(0);

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
rectangles = [];
for (x, y, w, h) in faces:
    rectangles.append(cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2))

print(len(rectangles))
cv2.imshow("Faces found", image)
cv2.waitKey(0)
