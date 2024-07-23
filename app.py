import cv2 as cv


def main():
    print("Запуск ...")
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv.CASCADE_SCALE_IMAGE)

        for x, y, w, h in faces:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv.imshow("pic", img)


        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
