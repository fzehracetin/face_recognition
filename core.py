# This is an example of using the k-nearest neighbors algorithm for face recognition
import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def exif_deleter(img_path):

    image = Image.open(img_path)  # EXIF bilgisi iceriyorsa silinecek
    try:
        image_exif = image._getexif()
        image_orientation = image_exif[274]

        print("Deleting EXIF metadata ...")

        # Rotate depending on orientation.
        if image_orientation == 2:  # 0 degrees, mirrored
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation == 3:  # 180 degrees
            image = image.transpose(Image.ROTATE_180)
        elif image_orientation == 4:  # 180 degrees and mirrored
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation == 5:  # 90 degrees
            image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.ROTATE_90)
        elif image_orientation == 6:  # 90 degrees, mirrored
            image = image.transpose(Image.ROTATE_270)
        elif image_orientation == 7:  # 270 degrees
            image = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.ROTATE_90)
        elif image_orientation == 8:  # 270 degrees, mirrored
            image = image.transpose(Image.ROTATE_90)

        # next 3 lines strip exif
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        image_without_exif.save(img_path)
    except:
        pass

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    x = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):

            exif_deleter(img_path)

            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                x.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)


    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    if model_save_path  is not None :
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)


def predict_image(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    print(model_path)
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    exif_deleter(X_img_path)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def predict_video(video_path, model_path, distance_threshold=0.6):
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    stream = cv2.VideoCapture(video_path)
    output_path = 'output.avi'
    fourcc = cv2.VideoWriter_fourcc('M','J', 'P', 'G')
    frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    ''
    # eger video dik cekildiyse ve dondurme varsa bu sekilde
    frame_width = int(stream.get(3))
    frame_height = int(stream.get(4))
    new_width = int(frame_width / 2)
    new_height = int(frame_height / 2)

    output = cv2.VideoWriter(output_path, fourcc, fps, (new_height, new_width))
    #eger videoda herhangi bir dondurma islemi yoksa ustteki commente kadar olan kisim commentlenir alt taraf commentten cikartilir''

    #output = cv2.VideoWriter(output_path, fourcc, fps, (int(stream.get(3)), int(stream.get(4))))

    print("frames : ", frames)

    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = stream.read()
        if not ret:
            pass
        frame_number +=1

        # eger video acilinca donuyorsa
        frame = ndimage.rotate(frame, 270)
        frame = cv2.resize(frame, (new_height, new_width))
        # -----------------yoksa commentle----------------------

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]


        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            print("yuz yok")
            output.write(frame)
            continue
        print("yuz var")
        # print("face locations : ", face_locations)

        faces_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                       zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]


        # Label the results
        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, frames))
        output.write(frame)

    # All done!
    stream.release()
    cv2.destroyAllWindows()

def predict_webcam(model_path, distance_threshold=0.6):
    video_capture = cv2.VideoCapture(0)

    process_this_frame = True

    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if len(face_locations) == 0:
                continue
            print("yuz var")
            faces_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

            predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                           zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]

            # Display the results
            for name, (top, right, bottom, left) in predictions:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        process_this_frame = not process_this_frame

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

