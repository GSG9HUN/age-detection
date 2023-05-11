import cv2
import dlib

img = cv2.imread('face.jpg')
img = cv2.resize(img, (720, 640))
# kép másolat.
imgCopy = img.copy()
# Már betanított NN model értékeinek beolvasása.
ageWeights = "pretrained_stuffs/age_deploy.prototxt"
ageConfig = "pretrained_stuffs/age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageConfig, ageWeights)

# Életkor lista.
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
modelMean = (78.4263377603, 87.7689143744, 114.895847746)

Boxes = []  # Az arcok keretezésére.
message = 'Eletkor'  # A szöveg ami megjelenik majd.

# Arc detektáló
face_detector = dlib.get_frontal_face_detector()
# szürtkítés
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# arcok keresése
faces = face_detector(img)

# Ha nem talál arcot rajta
if not faces:
    message = 'Nincsenek arcok a képen.'
    cv2.putText(img, f'{message}', (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 2)

else:
    # Az arcok bekeretezésére
    for face in faces:
        box = [face.left(), face.top(), face.right(), face.bottom()]
        Boxes.append(box)
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()),
                      (00, 200, 200), 2)

    for box in Boxes:
        #ide kell a képmásolatt mert a modell RGB-s képpel tud dolgozni, ezért nem jó neki a grayscale img.
        #kivágjuk csak az arcot.
        face = imgCopy[box[1]:box[3], box[0]:box[2]]
        # kép előkészítése.
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), modelMean, swapRB=False)

        # Életkor meghatározás.
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(img, f'{message}: {age}', (box[0],
                                             box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow("Eletkor meghatarozas", img)
cv2.waitKey(0)
