import os
import core


def train_model(train_dir, model_save_path, n_neighbors):
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    core.train(train_dir, model_save_path=model_save_path, n_neighbors=n_neighbors,)
    print("Training complete!")


def predict_image(test_dir, model_path):
    for image_file in os.listdir(test_dir):
        full_file_path = os.path.join(test_dir, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = core.predict_image(full_file_path, model_path=model_path)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        core.show_prediction_labels_on_image(os.path.join(test_dir, image_file), predictions)

def predict_video(video_path, model_path):
    print("Processing the video...")
    core.predict_video(video_path, model_path)
    print("Processing complete!")

def predict_web_cam(model_path):
    core.predict_webcam(model_path)

if __name__ == "__main__":

    # STEP 1: Train the KNN classifier and save it to disk

    train_dir = "images/train"
    model_save_path = "trained_knn_model.clf"
    n_neighbors = 2

    train_model(train_dir, model_save_path, n_neighbors)

    # STEP 2: Using the trained classifier, make predictions for unknown images

    test_dir = "images/test"
    model_path = "trained_knn_model.clf"
    predict_image(test_dir, model_path)

    # STEP 3: Using the trained classifier, make predictions for a video

    video_path = "videos/input6.mp4"
    #predict_video(video_path, model_path)

    # STEP 4: Using the trained classifier, make predictions for webcam

    # predict_web_cam(model_path)

