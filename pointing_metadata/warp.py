#!/usr/bin/env python
import sys
from os import path, mkdir
from lsst.afw import image as afwImage
from lsst.afw.math.warper import Warper
import pickle

diffim_topdirstr = "/gscratch/scrubbed/dinob/imdiff/{0:03d}/deepDiff"
save_topdirstr = "/gscratch/scrubbed/dinob/warps/{0:03d}/"

def warp_field_images(groupid):    
    group_id = int(groupid)

    with open('PickledPointings.pkl', 'rb') as f:
        Pointing_Groups = pickle.load(f)
    visit_num = Pointing_Groups[group_id]['visit_id']

    warper = Warper("lanczos4")

    for chip_num in range(1, 63):
        template_set = False

        for visit_id in visit_num:
            #                 topdir                               | visitdir | diffexp name     
            #/gscratch/scrubbed/dinob/imdiff/{groupid:03d}/deepDiff/v{visitid}/diffexp-{ccd:02d}.fits
            diffim_topdir = diffim_topdirstr.format(group_id)
            visit_dir = "v{0}/".format(visit_id)
            diffexp_name = "diffexp-{0:02d}.fits".format(chip_num)
            diffexp_path = path.join(diffim_topdir, visit_dir, diffexp_name)

            #                 save dir                  | warp name    
            #/gscratch/scrubbed/dinob/warps/{groupd:03d}/{0:03d}-{1:02d}.fits
            save_dir = save_topdirstr.format(group_id)
            warp_name = "{0:03d}-{1:02d}.fits".format(visit_id, chip_num)
            save_path = path.join(save_dir, warp_name)

            try:
                assert(path.exists(diffexp_path))
            except AssertionError:
                print("Assertion Error, path %s does not exist!" % diffexp_path)
                continue

            if not path.exists(save_dir):
                mkdir(save_dir)

            exp = afwImage.ExposureF(diffexp_path)

            if template_set is False:
                template = exp
                template_set = True
                exp.writeFits(save_path)
            else:
                warpedExp = warper.warpExposure(template.getWcs(), exp,
                                                destBBox=template.getBBox())
                warpedExp.writeFits(save_path)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        warp_field_images(sys.argv[1])
    else:
        raise TypeError(("Wrong number of arguments. " 
                         "Expected groupid, got {0}".format(sys.argv[1:])))
