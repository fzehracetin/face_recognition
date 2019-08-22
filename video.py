import face_recognition
import pickle
import cv2
from face_recognition.face_recognition_cli import image_files_in_folder
from scipy import ndimage

def video_processor (video_path, model_path, distance_threshold):

    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    stream = cv2.VideoCapture(video_path)
    output_path = 'output.avi'
    fourcc = cv2.VideoWriter_fourcc('M','J', 'P', 'G')
    frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    print(fps)

    frame_width = int(stream.get(3))
    frame_height = int(stream.get(4))

    new_width = int(frame_width / 2)
    new_height = int(frame_height / 2)

    print("frames : ", frames)

    # output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))
    output = cv2.VideoWriter(output_path, fourcc, fps, (new_height, new_width))
    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = stream.read()
        frame_number +=1
        # Quit when the input video file ends
        if not ret:
            break

        frame = ndimage.rotate(frame, 270)
        frame = cv2.resize(frame, (new_height, new_width))

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

        # print("faces encodings", faces_encodings)

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

if __name__== "__main__":

    video_processor("videos/input6.mp4", "trained_knn_model.clf", 0.6)