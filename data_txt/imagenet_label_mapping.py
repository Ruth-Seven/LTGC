import csv
import pathlib

lable2name = []
simple_classes = (
    "/home/hujunjie/tobacco-experiment/model/LTGC/data_txt/ImageNet_LT/ImageNet_cls_name.txt"
)

with open(pathlib.Path(simple_classes), "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        lable2name.append(row[0].split(",")[0])


def get_readable_name(label):
    name = lable2name[label]
    return name


print("test 998:" + get_readable_name(998))
print("test 123:" + get_readable_name(123))
print("test 77:" + get_readable_name(77))
