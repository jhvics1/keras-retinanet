import os
import argparse
import xml.etree.ElementTree

# /data/imgs/img_001.jpg,837,346,981,456,cow
# /data/imgs/img_002.jpg,215,312,279,391,cat
# /data/imgs/img_002.jpg,22,5,89,84,bird
# /data/imgs/img_003.jpg,,,,,

# < object >
#   < name > person < / name >
#   < pose > Unspecified < / pose >
#   < truncated > 0 < / truncated >
#   < difficult > 0 < / difficult >
#   < bndbox >
#       < xmin > 1029 < / xmin >
#       < ymin > 282 < / ymin >
#       < xmax > 1086 < / xmax >
#       < ymax > 436 < / ymax >
#   < / bndbox >
# < / object >


def transform_xml_to_csv(image_path, label_path):
    fout = open(label_path+'/../annotation.csv', 'wt')
    for path in sorted(os.listdir(label_path)):
        file_path = os.path.join(label_path, path)

        try:
            doc = xml.etree.ElementTree.parse(file_path)
            root = doc.getroot()
            # path = root.find('path').text.split('\\')
            # path = image_path + '/' + path[len(path) - 1]
            path = image_path + '/' + path.replace('xml', 'jpg')
            # print('path : {}'.format(path.text))

            for obj in root.findall('object'):
                name = obj.find('name')
                # print('name : {}'.format(name.text))
                box = obj.find('bndbox')
                transform = '{fname},{x1},{y1},{x2},{y2},{class_name}\n'.format(fname=path,
                                                                                x1=box.find('xmin').text,
                                                                                y1=box.find('ymin').text,
                                                                                x2=box.find('xmax').text,
                                                                                y2=box.find('ymax').text,
                                                                                class_name=name.text)
                fout.write(str(transform))
                fout.flush()
                # print(transform)
        except:
            print('Mal-formed file : {} - {}'.format(file_path, root))

    fout.close()


def print_args():
    parser = argparse.ArgumentParser('Runs 2 Stream inflated 3D ConvNet '
                                     '(based on inception v1 network) Model'
                                     'for Action Recognition')
    parser.add_argument('--images_path', type=str, default='.',
                        help='File path where all the images are stored')
    parser.add_argument('--label_path', type=str, default='./',
                        help='Directory path where all the label files are stored')

    return parser.parse_args()


if __name__ == "__main__":
    args = print_args()
    transform_xml_to_csv(args.images_path, args.label_path)
