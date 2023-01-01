import napari
from cellpose import models
from cellpose import io


def get_segmentation(model, img, is_show=False, diameter=0, flow_threshold=0.4, cellprob_threshold=0, save_path = None):
    
    
    model = models.CellposeModel(gpu=True, 
                             pretrained_model=model)

    # use model diameter if user diameter is 0
    diameter = model.diam_labels if diameter==0 else diameter


    pred = model.eval(img, 
                                  channels=[1,2],
                                  diameter=diameter,
                                  flow_threshold=flow_threshold,
                                  cellprob_threshold=cellprob_threshold
                                  )
    if is_show:
        viewer = napari.Viewer()
        phase = viewer.add_image(phase_img)
        mask = viewer.add_labels(masks)
        viewer.show(block=True)  # wait until viewer window closes

    return pred
