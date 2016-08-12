import cv2
import requests
import base64
import json

__author__ = 'Dandi Chen'

server = "http://detection.app.tusimple.sd/v1/analyzer/objdetect"


class BoundingBox(object):
    def __init__(self, top_left_x=0, top_left_y=0, bottom_right_x=0, bottom_right_y=0):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.width = self.bottom_right_x - self.top_left_x
        self.height = self.bottom_right_y - self.top_left_y

    def get_box_from_video(self, video_path, output_path):
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))

        video = cv2.VideoCapture(video_path)
        num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_count in range(num_of_frames):
            try:
                okay, frame = video.read()
                if not okay:
                    break
                binary = cv2.imencode('.jpg', frame)[1].tostring()
                encoded_string = base64.b64encode(binary)
                payload = {'image_base64': encoded_string, 'trim_detect': 0.8}
                response = requests.post(server, json=payload)
                result = json.loads(response.text)
                print result
                box = result['objs']
                output = frame.copy()
                self.vis_box(box, output, writer)
                # cv2.imshow('frame', output)
                # cv2.waitKey(0)
            except KeyboardInterrupt:
                print "can not read video " + video_path
                exit()

    def get_box_from_image(self, img_path, writer):
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        binary = cv2.imencode('.png', img)[1].tostring()
        encoded_string = base64.b64encode(binary)
        payload = {'image_base64': encoded_string, 'trim_detect': 0.8}
        response = requests.post(server, json=payload)
        result = json.loads(response.text)
        print result
        box = result['objs']
        output = img.copy()
        self.vis_box(box, output, writer, width, height, 1)

        box_edge = []
        for i in range(len(box)):
            # top_left_x, top_left_y, bom_right_x, bom_right_y
            box_edge.append([width * box[i]['left'], height * box[i]['top'],
                             width * box[i]['right'], height * box[i]['bottom']])
        return box_edge

    def vis_box(self, box, output, writer, width=1920, height=1080, flag=0):
        for i in range(len(box)):
            # draw bounding box
            bbox_type = box[i].get('type', '')
            if bbox_type == 'true_positive':
                color = (0, 255, 0)
            elif bbox_type == 'false_positive':
                color = (0, 0, 255)
            elif bbox_type == 'false_negative':
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            # top_left_x, top_left_y, bom_right_x, bom_right_y
            cv2.rectangle(output, (int(width*box[i]['left']), int(height*box[i]['top'])),
                          (int(width*box[i]['right']), int(height*box[i]['bottom'])), color, 4)

            # # # draw bounding box ID
            # text = str(box[i]['id'])
            # position = (box[i]['left'], box[i]['top'])
            # cv2.putText(output, text, position, cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
            print width*box[i]['left'], height*box[i]['top'], width*box[i]['right'], height*box[i]['bottom']

            cv2.imshow('box', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # write video
        if flag == 0:
            writer.write(output)
        else:
            cv2.imwrite(writer, output)

    def vis_box_grid(self, img, x_trans, y_trans, x_num, y_num, x_blk_size, y_blk_size, width, height,
                     img_x_trans=0, img_y_trans=0, box_x_trans=0, box_y_trans=0):
        for i in range(x_num - 1):
            cv2.line(img, (x_trans[i] + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
                     (x_trans[i] + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))
        cv2.line(img, (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
                 (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))

        for j in range(y_num - 1):
            cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                     (width + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans), (0, 255, 0))
        cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans),
                 (width + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans), (0, 255, 0))

        # cv2.imshow('box grid', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('box grid')

    def write_box_grid(self, img, x_trans, y_trans, x_num, y_num, x_blk_size, y_blk_size, width, height,
                      img_x_trans=0, img_y_trans=0, box_x_trans=0, box_y_trans=0):
        for i in range(x_num - 1):
            cv2.line(img, (x_trans[i] + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
                     (x_trans[i] + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))
        cv2.line(img, (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
                 (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))

        for j in range(y_num - 1):
            cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                     (width + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans), (0, 255, 0))
        cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans),
                 (width + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans), (0, 255, 0))

