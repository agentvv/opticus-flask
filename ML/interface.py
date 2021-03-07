from .HRNet_Facial_Landmark_Detection.lib.models import get_face_alignment_net
from .HRNet_Facial_Landmark_Detection.lib.models.hrnet import HighResolutionNet
from .HRNet_Facial_Landmark_Detection.lib.config import config
from .HRNet_Facial_Landmark_Detection.lib.utils.transforms import crop
import torch
import numpy as np
from PIL import Image
from skimage.transform import resize

from matplotlib import pyplot as plt
import cv2


#get predictions from score maps in torch Tensor
#return type: torch.LongTensor
def get_preds(scores):
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def getFaceMeasurements(f):
    config.defrost()
    config.merge_from_file("./ML/face_alignment_300w_hrnet_w18.yaml")
    config.OUTPUT_DIR = './ML/out'
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = get_face_alignment_net(config)
    device = torch.device('cpu')
    model.to(device)
    model = model.cpu()
    state_dict = torch.load("./ML/cpu_model.pth", map_location=device)
    model.load_state_dict(torch.load("./ML/cpu_model.pth", map_location=device))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.array(Image.open(f).convert('RGB'), dtype=np.float32)
    img = resize(img, [256,256])
    img = img.astype(np.float32)
    img = (img/255.0 - mean) / std
    img = img.transpose([2, 0, 1])
    output = model(torch.Tensor(img).unsqueeze(dim=0))
    output = output.data.cpu()
    coords = get_preds(output)
    coords = coords.squeeze()
    coords = coords.numpy()
    temple_points = [coords[0], coords[16]]
    eye_points = [coords[39], coords[42]]


    device = torch.device('cpu')
    fcn_load = models.segmentation.fcn_resnet101(pretrained=False)
    fcn_load.classifier = FCNHead(2048,2)
    fcn_load.aux_classifier = FCNHead(1024,2)
    #state_dict = torch.load("./paper_model", map_location=device)
    fcn_load.load_state_dict(torch.load("./paper_model", map_location=device))

    base = './paper_dataset/20210307_003056 (2).jpg'
    fcn_load.aux_classifier = None
    trf_x = T.Compose([T.Resize([256,256]),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    out = trf_x(Image.open(os.path.join(base)).convert('RGB'))
    out= out.unsqueeze(0)
    out = fcn_load(out.float())['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    om_list = list(om)
    for i in range(len(om_list)):
        if 0 in list(om_list[i]):
            top = i
            print(top)
            break
    for j in range(len(om_list)):
        if 0 in list(om_list[len(om_list)-1 - j ]):
            print(len(om_list)-1 - j)
            bottom = len(om_list)-1 - j 
            break
    distance  = bottom-top
    dist_eye = np.linalg.norm(eye_points[0]-eye_points[1])*4
    dist_temple =  np.linalg.norm(temple_points[0]-temple_points[1])*4
    out_eye = 11.5 * ((dist_eye*0.5)/distance)
    out_temple = 11.5 * (dist_temple/distance)
    faceMeasurementsCoords = [int(out_temple),int(out_eye)]
    #faceMeasurementsCoords = [int(( (abs(coords[16][0] - coords[ 0][0]))**2 + (abs(coords[16][1] - coords[ 0][1]))**2 )**(1/2)),
    #                          int(( (abs(coords[42][0] - coords[39][0]))**2 + (abs(coords[42][1] - coords[39][1]))**2 )**(1/2))]

    # ~ res = cv2.resize(img[2,:,:], dsize=(64, 64), interpolation=cv2.INTER_CUBIC )
    # ~ plt.imshow(res,cmap="cividis")
    # ~ plt.scatter(coords[:,0],coords[:,1])
    # ~ plt.savefig('foo.png')

    return faceMeasurementsCoords

