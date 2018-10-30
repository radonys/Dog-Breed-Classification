import os
import csv
import zipfile
import shutil

if os.path.exists('all/'):

	with open('all/labels.csv', mode='r') as csv_file:

		csv_reader = csv.DictReader(csv_file)
		line_count = 0
		directory = 'all/train_images/'

		if not os.path.exists(directory):
			os.makedirs(directory)


		for row in csv_reader:

			if line_count == 0:

				print('Column names are', {", ".join(row)})
				line_count += 1

			id_value = str(row["id"])
			breed = str(row["breed"])

			if not os.path.exists(os.path.join(directory, breed)):
				os.makedirs(os.path.join(directory, breed))

			traindir = 'all/train/'

			if not os.path.exists(traindir):

				zip_ref = zipfile.ZipFile('all/train.zip', 'r')
				zip_ref.extractall('all/')
				zip_ref.close()

			prev_dir = str(os.path.join(traindir, id_value + '.jpg'))
			new_dir = str(os.path.join(directory, breed, id_value + '.jpg'))
				
			if os.path.exists(os.path.join(traindir, id_value + '.jpg')):
				shutil.move(prev_dir, new_dir)

			line_count +=1

		print('Processed lines: ', line_count)
		os.rmdir(traindir)
		os.rename('all/train_images', 'all/train')

else:

	print('Data directory not found.')





