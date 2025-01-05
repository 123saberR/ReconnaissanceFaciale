import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(dataset_path):
    known_encodings = []
    known_names = []
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
        
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
    
    return known_encodings, known_names

def process_video_stream(known_encodings, known_names):
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error reading video")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            if matches:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    dataset_path = "checkPersonFolder"
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces(dataset_path)
    print("Database loaded successfully.")
    
    process_video_stream(known_encodings, known_names)

if __name__ == "__main__":
    main()
