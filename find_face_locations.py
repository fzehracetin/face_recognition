import face_recognition
from PIL import Image

def find_face_locations(image_path, name):

    image = face_recognition.load_image_file(image_path)

    face_landmarks_list = face_recognition.face_landmarks(image)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    if len(face_landmarks_list) == 0:

        image = Image.open(image_path)
        try:
            image_exif = image._getexif()
            image_orientation = image_exif[274]
            print("Image orientation : ", image_orientation)

            # Rotate depending on orientation.
            if image_orientation == 2: # 0 degrees, mirrored
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif image_orientation == 3: # 180 degrees
                image = image.transpose(Image.ROTATE_180)
            elif image_orientation == 4: # 180 degrees and mirrored
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation == 5: # 90 degrees
                image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.ROTATE_90)
            elif image_orientation == 6: # 90 degrees, mirrored
                image = image.transpose(Image.ROTATE_270)
            elif image_orientation == 7: # 270 degrees
                image = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.ROTATE_90)
            elif image_orientation == 8: # 270 degrees, mirrored
                image = image.transpose(Image.ROTATE_90)
        except:
            pass

        # next 3 lines strip exif
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)

        image_without_exif.save("images/" + name + ".jpg")

        image = face_recognition.load_image_file("images/" + name + ".jpg")

        face_landmarks_list = face_recognition.face_landmarks(image)

        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

if __name__ == "__main__":
    name = "faz"
    image_path = "images/" + name + ".jpg"
    find_face_locations(image_path, name)
