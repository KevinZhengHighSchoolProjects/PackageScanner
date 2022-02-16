# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="img1.jpg",
	help="path to input image to be OCR'd")
ap.add_argument("-c", "--min-conf", type=int, default=0,
	help="mininum confidence value to filter weak text detection")
args = vars(ap.parse_args())

# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pytesseract.image_to_data(rgb)
print(results)
min_conf = 80

# loop over each of the individual text localizations
for x, b in enumerate(results.splitlines()):
        if x != 0:
                b = b.split()
                print(b)
                if len(b)==12:
                        if float(b[10]) > min_conf:
                                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                                cv2.rectangle(image, (x, y), (w+x, h+y), (0, 0, 255), 3)
                                cv2.putText(image, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255),2)
# show the output image
cv2.imshow("Image", image)
cv2.imshow("RGB", rgb)
cv2.waitKey(0)
