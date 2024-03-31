import cv2
import os
import csv

std_name = input('Enter your name: ')
std_num = int(input('Enter num: '))
std_section = input('Enter section: ')
str_num = str(std_num)


parent_directory = 'C:\Projects\FRAS\DAMS\Databases'

directory = std_section

path = os.path.join(parent_directory, directory)

csv_file_path = f'{path}\{std_section}.csv'

name_exists = False

with open(csv_file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if row and row[0] == str_num and row[1] == std_name:
            name_exists = True
            break

if name_exists:
    print(f'Name "{std_name}" with number {std_num} already exists in the CSV file for section "{std_section}".')
else:
    # If the name and number are not in the CSV file, add them
    with open(csv_file_path, 'a', newline="\n") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([std_num, std_name])
        print(f'Name "{std_name}" with number {std_num} added to the CSV file for section "{std_section}".')


try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s" % path)

# Try catch block to check if a person's individual name's directory exists inside the path or not
try:
    os.mkdir(os.path.join(path, std_name+'.'+str(std_num)))
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s" % path)
yml_directory1 = os.path.join(f'C:\Projects\FRAS\DAMS\Databases\{std_section}','Files')
yml_files1=False
for f in os.listdir(yml_directory1):
    if f==f'{std_name}.yml':
        yml_files1=True
def generate_image():
    cam = cv2.VideoCapture(0)
    harcascadePath = "C:\Projects\FRAS\haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break  # Exit if the camera capture fails

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            sampleNum += 1

            # Save each student's image
            img_name = f"{path}/{std_name}.{str_num}/{sampleNum}.jpg"
            print(img_name)
            cv2.imwrite(img_name, gray[y:y+h, x:x+w])
            cv2.imshow('Face Recognition', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > 60:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Print student details saved with ID and Full Name
    res = f'Student details saved with ID: {std_num} and Full Name: {std_name}'
    print(res)
if (yml_files1):
    print("Student already exists")
else:
    generate_image()
paths = f'{path}/{std_name}.{str_num}'

# print(paths)
