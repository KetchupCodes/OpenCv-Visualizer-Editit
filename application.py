from flask import Flask, request ,render_template
import os
import cv2
import numpy as np
from guidedfilter import guided_filter
import glob

from matplotlib import pyplot as plt
 
application=Flask(__name__)

#Low Light Enhancement Modules
def get_illumination_channel(I, w):
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])

    return darkch, brightch

def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)
    flatbright = brightch.ravel()

    searchidx = (-flatbright).argsort()[:int(M*N*p)]
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c)
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t))

def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im3 = np.empty(I.shape, I.dtype);
    for ind in range(0, 3):
        im3[:, :, ind] = I[:, :, ind] / A[ind]
    dark_c, _ = get_illumination_channel(im3, w)
    dark_t = 1 - omega*dark_c
    corrected_t = init_t
    diffch = brightch - darkch

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i, j] < alpha):
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)

def get_final_image(I, A, refined_t, tmin):
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A

    return (J - np.min(J))/(np.max(J) - np.min(J))

def dehaze(I, tmin, w, alpha, omega, p, eps, reduce=False):
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright) 
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w, eps)
    J_refined = get_final_image(I, A, refined_t, tmin)
    
    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced

def reduce_init_t(init_t):
    init_t = (init_t*255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    init_t = cv2.LUT(init_t, table)
    init_t = init_t.astype(np.float64)/255
    return init_t

#End of LLE Functions

database={'ramu':'123','soni':'465'}
UPLOAD_FOLDER='static/photos'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@application.route('/')
def home1():
    return render_template('home.html')

@application.route()
def start():
    return render_template('home.html')

@application.route('/home')
def home():
    return render_template('home.html')


@application.route('/grayscale/upload-image', methods=['GET', 'POST'])
def uploadimage11():
        files = glob.glob('static/photos/*')
        for f in files:
            os.remove(f)
        if request.method=="POST" :
            image_file = request.files["image"]
            if image_file:
                image_location=os.path.join(application.config['UPLOAD_FOLDER'],image_file.filename)
                image_file.save(image_location)
                im = cv2.imread(image_location)
                orig = im.copy()

                
                f_enhanced2 =  cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                
                
                
                cv2.imwrite(os.path.join( application.config['UPLOAD_FOLDER'], 'final.jpg'), f_enhanced2)
                
                
                return render_template('upload_image.html',uploaded_image=image_file.filename,final_image='final.jpg',name_effect="grayscale")   
        return render_template('upload_image.html',name_effect="grayscale")

   
@application.route('/smoothen/upload-image', methods=['GET', 'POST'])
def uploadimage1122():
        files = glob.glob('static/photos/*')
        for f in files:
            os.remove(f)
        if request.method=="POST" :
            image_file = request.files["image"]
            if image_file:
                image_location=os.path.join(application.config['UPLOAD_FOLDER'],image_file.filename)
                image_file.save(image_location)
                im = cv2.imread(image_location)
                orig = im.copy()

                
                f_enhanced2 =cv2.bilateralFilter(im,9,75,75)
                
                
                
                cv2.imwrite(os.path.join( application.config['UPLOAD_FOLDER'], 'final.jpg'), f_enhanced2)
                
                
                return render_template('upload_image.html',uploaded_image=image_file.filename,final_image='final.jpg',name_effect="smoothen")   
        return render_template('upload_image.html',name_effect="smoothen")

    
    
    
    

@application.route('/lle/upload-image', methods=['GET', 'POST'])
def uploadimage():
        files = glob.glob('static/photos/*')
        for f in files:
            os.remove(f)
        if request.method=="POST" :
            image_file = request.files["image"]
            if image_file:
                image_location=os.path.join(application.config['UPLOAD_FOLDER'],image_file.filename)
                image_file.save(image_location)
                im = cv2.imread(image_location)
                orig = im.copy()

                tmin = 0.1   # minimum value for t to make J image
                w = 15       # window size, which determine the corseness of prior images
                alpha = 0.4  # threshold for transmission correction
                omega = 0.75 # this is for dark channel prior
                p = 0.1      # percentage to consider for atmosphere
                eps = 1e-3   # for J image

                I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
                I = I[:, :, :3] / 255

                f_enhanced = dehaze(I, tmin, w, alpha, omega, p, eps)
                f_enhanced2 = dehaze(I, tmin, w, alpha, omega, p, eps, True)
                
                
                cv2.imwrite(os.path.join( application.config['UPLOAD_FOLDER'], 'final.jpg'), f_enhanced2)
                
                
                return render_template('upload_image.html',uploaded_image=image_file.filename,final_image='final.jpg',name_effect="lle")   
        return render_template('upload_image.html',name_effect="lle")



@application.route('/grab-cut/upload-image', methods=['GET', 'POST'])
def uploadimage112():
        files = glob.glob('static/photos/*')
        for f in files:
            os.remove(f)
        if request.method=="POST" :
            image_file = request.files["image"]
            if image_file:
                image_location=os.path.join(application.config['UPLOAD_FOLDER'],image_file.filename)
                image_file.save(image_location)
                img = cv2.imread(image_location)

                                
                mask = np.zeros(img.shape[:2],np.uint8)
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (150,110,450,290)
                cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                img2 = img*mask2[:,:,np.newaxis]
                                
                
                
                cv2.imwrite(os.path.join( application.config['UPLOAD_FOLDER'], 'final.jpg'), img2)
                
                
                return render_template('upload_image.html',uploaded_image=image_file.filename,final_image='final.jpg',name_effect="grab-cut")   
        return render_template('upload_image.html',name_effect="grab-cut")


@application.route('/edge-detection/upload-image', methods=['GET', 'POST'])
def uploadimageED():
        files = glob.glob('static/photos/*')
        for f in files:
            os.remove(f)
        if request.method=="POST" :
            image_file = request.files["image"]
            if image_file:
                image_location=os.path.join(application.config['UPLOAD_FOLDER'],image_file.filename)
                image_file.save(image_location)
                img = cv2.imread(image_location)

               
                edges = cv2.Canny(img,100,200)               
                
                
                cv2.imwrite(os.path.join( application.config['UPLOAD_FOLDER'], 'final.jpg'), edges)
                
                
                return render_template('upload_image.html',uploaded_image=image_file.filename,final_image='final.jpg',name_effect="edge-detection")   
        return render_template('upload_image.html',name_effect="edge-detection")
    
    
    
    
@application.route('/image-erosion/upload-image', methods=['GET', 'POST'])
def uploadimageER():
        files = glob.glob('static/photos/*')
        for f in files:
            os.remove(f)
        if request.method=="POST" :
            image_file = request.files["image"]
            if image_file:
                image_location=os.path.join(application.config['UPLOAD_FOLDER'],image_file.filename)
                image_file.save(image_location)
                img = cv2.imread(image_location)

               
                kernel = np.ones((5,5),np.uint8)

                edges = cv2.erode(img, kernel, iterations=1)

                          
                
                
                cv2.imwrite(os.path.join( application.config['UPLOAD_FOLDER'], 'final.jpg'), edges)
                
                
                return render_template('upload_image.html',uploaded_image=image_file.filename,final_image='final.jpg',name_effect="image-erosion")   
        return render_template('upload_image.html',name_effect="image-erosion")    



@application.route('/photos/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(application.config['UPLOAD_FOLDER'], filename)



if __name__ == "__main__":
    application.run(debug=True)