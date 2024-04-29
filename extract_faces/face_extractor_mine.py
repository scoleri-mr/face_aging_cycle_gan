# Import libraries
import os
import cv2
import numpy as np
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Extract face images")
	parser.add_argument("--input_folder", type=str, default='images', help="extract faces from images in this folder")
	parser.add_argument("--output_folder", type=str, default='faces', help="save extracted faces as images in this folder")
	parser.add_argument("--original", type=int, default = 0, help='set to 1 if you want to use the original code')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	# Define paths, the model has to be in the same folder as this file
	base_dir = os.path.dirname(__file__) +'/'
	prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
	caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

	# Read the model
	model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

	# Create output directory if it does not exist
	if not os.path.exists(args.output_folder):
		print("New directory created")
		os.makedirs(args.output_folder)

	# Loop through all images and strip out faces
	count = 0
	for n, file in enumerate(os.listdir(args.input_folder)):
		if n%50==0 and n!=0:
			print(f'Extracted {n} images...')
			
		file_name, file_extension = os.path.splitext(file)
		if (file_extension in ['.png','.jpg']):
			image = cv2.imread(args.input_folder + file)

			(h, w) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			model.setInput(blob)
			detections = model.forward()

			# Identify each face
			for i in range(0, detections.shape[2]):
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				confidence = detections[0, 0, i, 2]

				# If confidence > 0.5, save it as a separate file
				if (confidence > 0.5):
					
					if(args.original==0):
						centroX = int((endX-startX)/2 + startX)
						centroY = int((endY-startY)/2 + startY)
						l = int(max(endX - startX, endY - startY))*1.3
						l = int(l/2)
						startX = max(0, centroX - l)
						startY = max(0, centroY - l)
						endX = min(w, centroX + l)
						endY = min(h, centroY + l)

					count += 1
					frame = image[startY:endY, startX:endX]
					if frame is None:
						print(f'frame none, count={count}')
					try:
						cv2.imwrite(args.output_folder + str(i) + '_' + file, frame)
					except:
						continue
					
					break
	print("Extracted " + str(count) + " faces from all images")