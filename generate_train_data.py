import os
import random
import shutil

training = "./DataSet/training/"
validatePath = "./DataSet/validate/"
all_items = []

def copy_file_to_validate(number_of_copy,list_of_items,directory_from_copy,directory_to_copy):
    for j in range(0 ,number_of_copy):
        item_to_copy = random.choice(list_of_items)
        list_of_items.remove(item_to_copy)
        shutil.move(directory_from_copy+"/"+item_to_copy,directory_to_copy)


for i in range(0,100):
    if i < 10:
        directory= training+"0"+str(i)
        all_items = os.listdir(directory)
        twentyPercent = int(len(all_items)*0.2)
        copy_file_to_validate(twentyPercent,all_items,directory_from_copy=directory,directory_to_copy=validatePath+"0"+str(i))

    else:
        directory= training+str(i)
        all_items = os.listdir(directory)
        twentyPercent = int(len(all_items) * 0.2)
        copy_file_to_validate(twentyPercent,all_items,directory_from_copy=directory,directory_to_copy=validatePath+str(i))


