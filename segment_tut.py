import cv2
import numpy as np
import dicom
import json
import os
import random
import re
import shutil
import sys
from matplotlib import image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


#
# PARAMETERS
#


# number of bins to use in histogram for gaussian regression
NUM_BINS = 100
# number of standard deviations past which we will consider a pixel an outlier
STD_MULTIPLIER = 2
# number of points of our interpolated dataset to consider when searching for
# a threshold value; the function by default is interpolated over 1000 points,
# so 250 will look at the half of the points that is centered around the known
# myocardium pixel
THRESHOLD_AREA = 250
# number of pixels on the line within which to search for a connected component
# in a thresholded image, increase this to look for components further away
COMPONENT_INDEX_TOLERANCE = 20
# number of angles to search when looking for the correct orientation
ANGLE_SLICES = 36


#
# FUNCTIONS
#


def log(msg, lvl):
    string = ""
    for i in range(lvl):
        string += " "
    string += msg
    print string


def auto_segment_all_datasets(segment_fn=None, prefix="", ns=None, basepath=None, train=True, validate=True):

    d = basepath
    if d is None:
        d = sys.argv[1]

    studies = None

    if train and validate:
        log("Processing training and validation data",0)
        studies = next(os.walk(os.path.join(d, "train")))[1] + \
                  next(os.walk(os.path.join(d, "validate")))[1]
    elif train and not validate:
        log("Processing training data only",0)
        studies = next(os.walk(os.path.join(d, "train")))[1]
    elif not train and validate:
        log("Processing validation data only",0)
        studies = next(os.walk(os.path.join(d, "validate")))[1]
    else:
        log("Nothing to do.",0)
        return

    labels = np.loadtxt(os.path.join(d, "train.csv"), delimiter=",",
                        skiprows=1)

    label_map = {}
    for l in labels:
        label_map[l[0]] = (l[2], l[1])

    num_samples = ns
    if num_samples is None:
        if len(sys.argv) > 2:
            num_samples = int(sys.argv[2])
            log("Processing {} samples".format(num_samples),0)
            studies = random.sample(studies, num_samples)
        else:
            log("Processing all samples",0)
    else:
        num_samples = int(num_samples)
        log("Processing {} samples".format(num_samples),0)
        studies = random.sample(studies, num_samples)

    if segment_fn is None:
        segment_fn = segment_dataset

    path = "{}output".format(prefix)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    comp_file = "{}completed.json".format(prefix)
    comp_json = None
    comp_sets = []

    if os.path.isfile(comp_file):
        comp_json = open(comp_file,"r+")
        comp_sets = json.load(comp_json)
    else:
        comp_json = open(comp_file, "w")

    accu_file = "{}accuracy.csv".format(prefix)
    accuracy_csv = None

    if os.path.isfile(accu_file):
        accuracy_csv = open(accu_file, "a")
    else:
        accuracy_csv = open(accu_file, "w")
        accuracy_csv.write("Dataset,EDV_actual,EDV_predicted,EDV_diff,ESV_actual,ESV_predicted,ESV_diff,EF_actual,EF_predicted,EF_diff\n")

    submit_file = "{}submit.csv".format(prefix)
    submit_csv = None

    if os.path.isfile(submit_file):
        submit_csv = open(submit_file, "a")
    else:
        submit_csv = open(submit_file, "w")

        submit_csv.write("Id,")
        for i in range(0, 600):
            submit_csv.write("P%d" % i)
            if i != 599:
                submit_csv.write(",")
            else:
                submit_csv.write("\n")

    results = []
    for s in studies:

        probs_sys = []
        probs_dis = []

        if comp_sets and num_samples is None and int(s) in comp_sets:
            log("Dataset {} already processed".format(s),0)
            continue

        if int(s) <= 500:
            full_path = os.path.join(d, "train", s)
        else:
            full_path = os.path.join(d, "validate", s)

        dset = Dataset(full_path, s)
        (edv, esv) = label_map.get(int(dset.name), (None, None))
        p_edv = 0
        p_esv = 0

        if edv:
            probs_sys.append(esv)
            probs_dis.append(edv)

        try:
            dset.load()
            segment_fn(dset)
            if dset.edv >= 600 or dset.esv >= 600:
            #    raise Exception("Prediction too large")
                log("Prediction too large for dataset %s" % (dset.name),0)
            p_edv = dset.edv
            p_esv = dset.esv
        except Exception as e:
            log("***ERROR***: Exception %s thrown by dataset %s" % (str(e), dset.name), 0)
        submit_csv.write("%d_Systole," % int(dset.name))
        for i in range(0, 600):
            if i < p_esv:
                submit_csv.write("0.0")
                probs_sys.append(0.0)
            else:
                submit_csv.write("1.0")
                probs_sys.append(1.0)
            if i == 599:
                submit_csv.write("\n")
            else:
                submit_csv.write(",")
        submit_csv.write("%d_Diastole," % int(dset.name))
        for i in range(0, 600):
            if i < p_edv:
                submit_csv.write("0.0")
                probs_dis.append(0.0)
            else:
                submit_csv.write("1.0")
                probs_dis.append(1.0)
            if i == 599:
                submit_csv.write("\n")
            else:
                submit_csv.write(",")
        
        if edv is not None:
            results.append(probs_sys)
            results.append(probs_dis)

            ef_actual = (edv - esv)/edv
            ef_pred = (p_edv - p_esv)/p_edv
            accuracy_csv.write("%s,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (dset.name, edv, p_edv, edv-p_edv, esv, p_esv, esv-p_esv, ef_actual, ef_pred, ef_actual-ef_pred))

        comp_sets.append(int(s))
        comp_json.seek(0)
        comp_json.write(json.dumps(comp_sets))
        comp_json.flush()

    print "CRPS={}".format(CRPS_mean(results))
    
    comp_json.close()
    accuracy_csv.close()
    submit_csv.close()

class Dataset(object):
    dataset_count = 0

    def __init__(self, directory, subdir):
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs:
            m = re.match("sax_(\d+)", s)
            if m is not None:
                slices.append(int(m.group(1)))

        slices_map = {}
        first = True
        times = []
        bad_slices = set()

        for s in slices:
            files = next(os.walk(os.path.join(directory, "sax_%d" % s)))[2]
            offset = None
            found_bad_slice = False

            for f in files:
                if f.endswith(".dcm"):
                    m = re.match("IM-(\d{4,})-(\d{4})\.dcm", f)
                    if m is not None:
                        if first:
                            times.append(int(m.group(2)))
                        if offset is None:
                            offset = int(m.group(1))
                    else:
                        bad_slices.add(s)
                        found_bad_slice = True
                        break

            if found_bad_slice:
                continue
            else:
                first = False
                slices_map[s] = offset

        self.directory = directory
        self.time = sorted(times)
        self.slices = sorted(filter(lambda ss: ss not in bad_slices, slices))
        self.slices_map = slices_map
        Dataset.dataset_count += 1
        self.name = subdir

        if bad_slices:
            log("Found bad slices {} for dataset {}".format(list(bad_slices),self.name),0)

    def _filename(self, s, t):
        return os.path.join(self.directory,"sax_%d" % s, "IM-%04d-%04d.dcm" % (self.slices_map[s], t))

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array
        (x,y) = img.shape
        if x < y:
            img = img.transpose()
        return np.array(img)

    def _read_images_and_crop(self):

        def _crop_image(img):
            (y,x) = img.shape

            dy = (y - min_h)/2
            dx = (x - min_w)/2

            return img[dy:(y-dy),dx:(x-dx)]

        temp_images = [[self._read_dicom_image(self._filename(d, i))
                            for i in self.time]
                            for d in self.slices]

        times_r = range(len(self.time))
        slice_r = range(len(self.slices))

        [all_h,all_w] = zip(*[temp_images[i][j].shape for j in times_r for i in slice_r])
        min_h = min(all_h)
        min_w = min(all_w)

        return [[_crop_image(temp_images[i][j]) for j in times_r] for i in slice_r]

    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
            if dist == 0.0:
                raise AttributeError
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...
        
        self.images = np.array(self._read_images_and_crop())

        self.dist = dist
        self.area_multiplier = x * y

    def load(self):
        self._read_all_dicom_images()


# assumes dataset is loaded, call dataset.load()
def segment_dataset(dataset):
    images = dataset.images
    dist = dataset.dist
    areaMultiplier = dataset.area_multiplier
    # shape: num slices, num snapshots, rows, columns
    log("Calculating rois...", 1)
    rois, circles = calc_rois(images)
    log("Calculating areas...", 1)
    all_masks, all_areas = calc_all_areas(images, rois, circles)
    log("Calculating volumes...", 1)
    area_totals = [calc_total_volume(a, areaMultiplier, dist)
                   for a in all_areas]
    log("Calculating ef...", 1)
    edv = max(area_totals)
    esv = min(area_totals)
    ef = (edv - esv) / edv
    log("Done, ef is %f" % ef, 1)

    save_masks_to_dir(dataset, all_masks)

    output = {}
    output["edv"] = edv
    output["esv"] = esv
    output["ef"] = ef
    output["areas"] = all_areas.tolist()
    f = open("output/%s/output.json" % dataset.name, "w")
    json.dump(output, f, indent=2)
    f.close()
    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef


def save_masks_to_dir(dataset, all_masks):
    os.mkdir("output/%s" % dataset.name)
    for t in range(len(dataset.time)):
        os.mkdir("output/%s/time%02d" % (dataset.name, t))
        for s in range(len(dataset.slices)):
            mask = all_masks[t][s]
            image.imsave("output/%s/time%02d/slice%02d_mask.png" %
                         (dataset.name, t, s), mask)
            eroded = binary_erosion(mask)
            hollow_mask = np.where(eroded, 0, mask)
            colorimg = cv2.cvtColor(dataset.images[s][t],
                                    cv2.COLOR_GRAY2RGB)
            colorimg = colorimg.astype(np.uint8)
            colorimg[hollow_mask != 0] = [255, 0, 255]
            image.imsave("output/%s/time%02d/slice%02d_color.png" %
                         (dataset.name, t, s), colorimg)


def calc_rois(images):
    (num_slices, _, _, _) = images.shape
    log("Calculating mean...", 2)
    dc = np.mean(images, 1)

    def get_H1(i):
        log("Fourier transforming on slice %d..." % i, 3)
        ff = fftn(images[i])
        first_harmonic = ff[1, :, :]
        log("Inverse Fourier transforming on slice %d..." % i, 3)
        result = np.absolute(ifftn(first_harmonic))
        log("Performing Gaussian blur on slice %d..." % i, 3)
        result = cv2.GaussianBlur(result, (5, 5), 0)
        return result

    log("Performing Fourier transforms...", 2)
    h1s = np.array([get_H1(i) for i in range(num_slices)])
    m = np.max(h1s) * 0.05
    h1s[h1s < m] = 0

    log("Applying regression filter...", 2)
    regress_h1s = regression_filter(h1s)
    log("Post-processing filtered images...", 2)
    proc_regress_h1s, coords = post_process_regression(regress_h1s)
    log("Determining ROIs...", 2)
    rois, circles = get_ROIs(dc, proc_regress_h1s, coords)
    return rois, circles


def calc_all_areas(images, rois, circles):
    closest_slice = get_closest_slice(rois)
    (_, times, _, _) = images.shape

    def calc_areas(time):
        log("Calculating areas at time %d..." % time, 2)
        mask, mean = locate_lv_blood_pool(images, rois, circles, closest_slice,
                                          time)
        masks, areas = propagate_segments(images, rois, mask, mean,
                                          closest_slice, time)
        return (masks, areas)

    result = np.transpose(map(calc_areas, range(times)))
    all_masks = result[0]
    all_areas = result[1]
    return all_masks, all_areas


def calc_total_volume(areas, area_multiplier, dist):
    slices = np.array(sorted(areas.keys()))
    modified = [areas[i] * area_multiplier for i in slices]
    vol = 0
    for i in slices[:-1]:
        a, b = modified[i], modified[i+1]
        subvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += subvol / 1000.0  # conversion to mL
    return vol


def get_centroid(img):
    nz = np.nonzero(img)
    pxls = np.transpose(nz)
    weights = img[nz]
    avg = np.average(pxls, axis=0, weights=weights)
    return avg


def regress_centroids(cs):
    num_slices = len(cs)
    y_centroids = cs[:, 0]
    x_centroids = cs[:, 1]
    z_values = np.array(range(num_slices))

    (xslope, xintercept, _, _, _) = linregress(z_values, x_centroids)
    (yslope, yintercept, _, _, _) = linregress(z_values, y_centroids)

    return (xslope, xintercept, yslope, yintercept)


def get_weighted_distances(imgs, coords, xs, xi, ys, yi):
    a = np.array([0, yi, xi])
    n = np.array([1, ys, xs])

    zeros = np.zeros(3)

    def dist(p):
        to_line = (a - p) - (np.dot((a - p), n) * n)
        d = euclidean(zeros, to_line)
        return d

    def weight(p):
        (z, y, x) = p
        return imgs[z, y, x]

    dists = np.array([dist(c) for c in coords])
    weights = np.array([weight(c) for c in coords])
    return (coords, dists, weights)


def gaussian_fit(dists, weights):
    # based on http://stackoverflow.com/questions/11507028/fit-a-gaussian-function
    (x, y) = histogram_transform(dists, weights)
    fivep = int(len(x) * 0.05)
    xtmp = x
    ytmp = y
    fromFront = False
    while True:
        if len(xtmp) == 0 and len(ytmp) == 0:
            if fromFront:
                # well we failed
                idx = np.argmax(y)
                xmax = x[idx]
                p0 = [max(y), xmax, xmax]
                (A, mu, sigma) = p0
                return mu, sigma, lambda x: gauss(x, A, mu, sigma)
            else:
                fromFront = True
                xtmp = x
                ytmp = y

        idx = np.argmax(ytmp)
        xmax = xtmp[idx]

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        p0 = [max(ytmp), xmax, xmax]
        try:
            coeff, var_matrix = curve_fit(gauss, xtmp, ytmp, p0=p0)
            (A, mu, sigma) = coeff
            return (mu, sigma, lambda x: gauss(x, A, mu, sigma))
        except RuntimeError:
            if fromFront:
                xtmp = xtmp[fivep:]
                ytmp = ytmp[fivep:]
            else:
                xtmp = xtmp[:-fivep]
                ytmp = ytmp[:-fivep]


def histogram_transform(values, weights):
    hist, bins = np.histogram(values, bins=NUM_BINS, weights=weights)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + (bin_width / 2)

    return (bin_centers, hist)


def get_outliers(coords, dists, weights):
    fivep = int(len(weights) * 0.05)
    ctr = 1
    while True:
        (mean, std, fn) = gaussian_fit(dists, weights)
        low_values = dists < (mean - STD_MULTIPLIER*np.abs(std))
        high_values = dists > (mean + STD_MULTIPLIER*np.abs(std))
        outliers = np.logical_or(low_values, high_values)
        if len(coords[outliers]) == len(coords):
            weights[-fivep*ctr:] = 0
            ctr += 1
        else:
            return coords[outliers]


def regress_and_filter_distant(imgs):
    centroids = np.array([get_centroid(img) for img in imgs])
    raw_coords = np.transpose(np.nonzero(imgs))
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    (coords, dists, weights) = get_weighted_distances(imgs, raw_coords, xslope,
                                                      xintercept, yslope,
                                                      yintercept)
    outliers = get_outliers(coords, dists, weights)
    imgs_cpy = np.copy(imgs)
    for c in outliers:
        (z, x, y) = c
        imgs_cpy[z, x, y] = 0
    return imgs_cpy


def regression_filter(imgs):
    condition = True
    iternum = 0
    while(condition):
        log("Beginning iteration %d of regression..." % iternum, 3)
        iternum += 1
        imgs_filtered = regress_and_filter_distant(imgs)
        c1 = get_centroid(imgs)
        c2 = get_centroid(imgs_filtered)
        dc = np.linalg.norm(c1 - c2)
        imgs = imgs_filtered
        condition = (dc > 1.0)  # because python has no do-while loops
    return imgs


def post_process_regression(imgs):
    (numimgs, _, _) = imgs.shape
    centroids = np.array([get_centroid(img) for img in imgs])
    log("Performing final centroid regression...", 3)
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    imgs_cpy = np.copy(imgs)

    def filter_one_img(zlvl):
        points_on_zlvl = np.transpose(imgs[zlvl].nonzero())
        points_on_zlvl = np.insert(points_on_zlvl, 0, zlvl, axis=1)
        (coords, dists, weights) = get_weighted_distances(imgs, points_on_zlvl,
                                                          xslope, xintercept,
                                                          yslope, yintercept)
        outliers = get_outliers(coords, dists, weights)
        for c in outliers:
            (z, x, y) = c
            imgs_cpy[z, x, y] = 0

    log("Final image filtering...", 3)
    for z in range(numimgs):
        log("Filtering image %d of %d..." % (z+1, numimgs), 4)
        filter_one_img(z)

    return (imgs_cpy, (xslope, xintercept, yslope, yintercept))


def floats_draw_circle(img, center, r, color, thickness):
    (x, y) = center
    x, y = int(np.round(x)), int(np.round(y))
    r = int(np.round(r))
    cv2.circle(img, center=(x, y), radius=r, color=color, thickness=thickness)


def filled_ratio_of_circle(img, center, r):
    mask = np.zeros_like(img)
    floats_draw_circle(mask, center, r, 1, -1)
    masked = mask * img
    (x, _) = np.nonzero(mask)
    (x2, _) = np.nonzero(masked)
    if x.size == 0:
        return 0
    return float(x2.size) / x.size


def circle_smart_radius(img, center):
    domain = np.arange(1, 100)
    (xintercept, yintercept) = center

    def ratio(r):
        return filled_ratio_of_circle(img, (xintercept, yintercept), r)*r

    y = np.array([ratio(d) for d in domain])
    most = np.argmax(y)
    return domain[most]


def get_ROIs(originals, h1s, regression_params):
    (xslope, xintercept, yslope, yintercept) = regression_params
    (num_slices, _, _) = h1s.shape
    results = []
    circles = []
    for i in range(num_slices):
        log("Getting ROI in slice %d..." % i, 3)
        o = originals[i]
        h = h1s[i]
        ctr = (xintercept + xslope * i, yintercept + yslope * i)
        r = circle_smart_radius(h, ctr)
        tmp = np.zeros_like(o)
        floats_draw_circle(tmp, ctr, r, 1, -1)
        results.append(tmp * o)
        circles.append((ctr, r))

    return (np.array(results), np.array(circles))


def bresenham(x0, x1, y0, y1, fn):
    # using some pseudocode from
    # https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
    # and also https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    steep = abs(y1-y0) > abs(x1-x0)
    if steep:
        x0, x1, y0, y1 = y0, y1, x0, x1
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    def plot(x, y):
        if steep:
            fn(y, x)
        else:
            fn(x, y)

    dx = x1 - x0
    dy = y1 - y0

    D = 2*np.abs(dy) - dx
    plot(x0, y0)
    y = y0

    for x in range(x0+1, x1+1):  # x0+1 to x1
        D = D + 2*np.abs(dy)
        if D > 0:
            y += np.sign(dy)
            D -= 2*dx
        plot(x, y)


def line_thru(bounds, center, theta):
    (xmin, xmax, ymin, ymax) = bounds
    (cx, cy) = center

    if np.cos(theta) == 0:
        return (cx, ymin, cx, ymax)
    slope = np.tan(theta)

    x0 = xmin
    y0 = cy - (cx - xmin) * slope
    if y0 < ymin:
        y0 = ymin
        x0 = max(xmin, cx - ((cy - ymin) / slope))
    elif y0 > ymax:
        y0 = ymax
        x0 = max(xmin, cx - ((cy - ymax) / slope))

    x1 = xmax
    y1 = cy + (xmax - cx) * slope
    if y1 < ymin:
        y1 = ymin
        x1 = min(xmax, cx + ((ymin - cy) / slope))
    elif y1 > ymax:
        y1 = ymax
        x1 = min(xmax, cx + ((ymax - cy) / slope))

    return (x0, x1, y0, y1)


def get_line_coords(w, h, cx, cy, theta):
    coords = np.floor(np.array(line_thru((0, w-1, 0, h-1), (cx, cy), theta)))
    return coords.astype(np.int)


def trim_zeros_indices(has_zeros):
    first = 0
    for i in has_zeros:
        if i == 0:
            first += 1
        else:
            break

    last = len(has_zeros)
    for i in has_zeros[::-1]:
        if i == 0:
            last -= 1
        else:
            break

    return first, last


def get_line(roi, cx, cy, theta):
    (h, w) = roi.shape
    (x0, x1, y0, y1) = get_line_coords(w, h, cx, cy, theta)

    intensities = []
    coords = []

    def collect(x, y):
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        intensities.append(roi[y, x])
        coords.append((x, y))

    bresenham(x0, x1, y0, y1, collect)

    def geti(idx):
        return intensities[idx]

    getiv = np.vectorize(geti)
    x = np.arange(0, len(intensities))
    y = getiv(x)
    first, last = trim_zeros_indices(y)
    trimy = y[first:last]
    trimcoords = coords[first:last]

    trimx = np.arange(0, trimy.size)
    return (trimx, trimy, trimcoords)


def find_best_angle(roi, circ):
    ((cx, cy), r) = circ
    results = np.zeros(ANGLE_SLICES)
    fns = [None for i in range(ANGLE_SLICES)]

    COS_MATCHED_FILTER_FREQ = 2.5

    def score_matched(trimx, trimy):
        # first, normalize this data
        newtrimx = np.linspace(0.0, 1.0, np.size(trimx))
        minimum = np.min(trimy)
        maximum = np.max(trimy) - minimum
        newtrimy = (trimy - minimum) / maximum

        filt = 1 - ((np.cos(COS_MATCHED_FILTER_FREQ*2*np.pi*newtrimx)) /
                    2 + (0.5))
        cr = correlate(newtrimy, filt, mode="same")
        return np.max(cr)

    for i in range(ANGLE_SLICES):
        trimx, trimy, trimcoords = get_line(roi, cx, cy, np.pi*i/ANGLE_SLICES)
        score2 = score_matched(trimx, trimy)
        results[i] = score2
        fns[i] = (UnivariateSpline(trimx, trimy), trimx, trimcoords)

    best = np.argmax(results)
    return (best * np.pi / ANGLE_SLICES, fns[best])


def find_threshold_point(best, best_fn):
    fn, trimx, trim_coords = best_fn
    dom = np.linspace(np.min(trimx), np.max(trimx), 1000)
    f = fn(dom)
    mins = argrelmin(f)

    closest_min = -1
    closest_dist = -1
    for m in np.nditer(mins):
        dist = np.abs(500 - m)
        if closest_min == -1 or closest_dist > dist:
            closest_min = m
            closest_dist = dist

    fnprime = fn.derivative()
    restrict = dom[np.max(closest_min-THRESHOLD_AREA, 0):
                   closest_min+THRESHOLD_AREA]
    f2 = fnprime(restrict)

    m1 = restrict[np.argmax(f2)]
    mean = fn(m1)

    idx = np.min([int(np.floor(m1))+1, len(trim_coords)-1])
    return (mean, trim_coords, idx)


def get_closest_slice(rois):
    ctrd = get_centroid(rois)
    closest_slice = int(np.round(ctrd[0]))
    return closest_slice


def locate_lv_blood_pool(images, rois, circles, closest_slice, time):
    best, best_fn = find_best_angle(rois[closest_slice],
                                    circles[closest_slice])
    mean, coords, idx = find_threshold_point(best, best_fn)
    thresh, img_bin = cv2.threshold(images[closest_slice,
                                           time].astype(np.float32),
                                    mean, 255.0, cv2.THRESH_BINARY)
    labeled, num = label(img_bin)
    x, y = coords[idx]

    count = 0
    # Look along the line for a component. If one isn't found within a certain
    # number of indices, just spit out the original coordinate.
    while labeled[y][x] == 0:
        idx += 1
        count += 1
        x, y = coords[idx]
        if count > COMPONENT_INDEX_TOLERANCE:
            idx -= count
            x, y = coords[idx]
            break

    if count <= COMPONENT_INDEX_TOLERANCE:
        component = np.transpose(np.nonzero(labeled == labeled[y][x]))
    else:
        component = np.array([[y, x]])

    hull = cv2.convexHull(component)
    squeezed = hull
    if count <= COMPONENT_INDEX_TOLERANCE:
        squeezed = np.squeeze(squeezed)
    hull = np.fliplr(squeezed)

    #mask = np.zeros_like(labeled, dtype=np.float32)
    mask = np.zeros_like(labeled)
    cv2.drawContours(mask, [hull], 0, 255, thickness=-1)
    return mask, mean


def propagate_segments(images, rois, base_mask, mean, closest_slice, time):
    def propagate_segment(i, mask):
        thresh, img_bin = cv2.threshold(images[i,
                                               time].astype(np.float32),
                                        mean, 255.0, cv2.THRESH_BINARY)

        labeled, features = label(img_bin)

        region1 = mask == 255
        max_similar = -1
        max_region = 0
        for j in range(1, features+1):
            region2 = labeled == j
            intersect = np.count_nonzero(np.logical_and(region1, region2))
            union = np.count_nonzero(np.logical_or(region1, region2))
            similar = float(intersect) / union
            if max_similar == -1 or max_similar < similar:
                max_similar = similar
                max_region = j

        if max_similar == 0:
            component = np.transpose(np.nonzero(mask))
        else:
            component = np.transpose(np.nonzero(labeled == max_region))
        hull = cv2.convexHull(component)
        hull = np.squeeze(hull)
        if hull.shape == (2L,):
            hull = np.array([hull])
        hull = np.fliplr(hull)

        newmask = np.zeros_like(img_bin)

        cv2.drawContours(newmask, [hull], 0, 255, thickness=-1)

        return newmask

    (rois_depth, _, _) = rois.shape
    newmask = base_mask
    masks = {}
    areas = {}
    masks[closest_slice] = base_mask
    areas[closest_slice] = np.count_nonzero(base_mask)
    for i in range(closest_slice-1, -1, -1):
        newmask = propagate_segment(i, newmask)
        masks[i] = newmask
        areas[i] = np.count_nonzero(newmask)

    newmask = base_mask
    for i in range(closest_slice+1, rois_depth):
        newmask = propagate_segment(i, newmask)
        masks[i] = newmask
        areas[i] = np.count_nonzero(newmask)

    return masks, areas

def CRPS_mean(results):

    def CRPS_row(row):
        v_m = row[0]
        p = np.array(row[1:])
        v = np.array(range(len(p)))
        h = v >= v_m
        sq_dists = (p - h)**2
        return (np.sum(sq_dists)/len(sq_dists))

    crps_mean = 0.0
    if len(results) > 0:
        crps_vec = [CRPS_row(r) for r in results]
        crps_sum = np.sum(crps_vec)
        crps_mean = crps_sum/len(crps_vec)

    return crps_mean

if __name__ == "__main__":
    random.seed()
    auto_segment_all_datasets()
