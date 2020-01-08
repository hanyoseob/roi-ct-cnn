Paper
===============
* One network to solve all ROIs: Deep learning CT for any ROI using differentiated backprojection
  * Accepted by Medical Physics: [https://arxiv.org/abs/1810.00500]

Implementation
===============
* MatConvNet (matconvnet-1.0-beta24)
  * Please run the matconvnet-1.0-beta24/matlab/vl_compilenn.m file to compile matconvnet.
  * There is instruction on "http://www.vlfeat.org/matconvnet/mfiles/vl_compilenn/"
* One Network to Solve All ROIs (matconvnet-1.0-beta24/examples/interior-tomography)
  * Please run the matconvnet-1.0-beta24/examples/interior-tomography/download_pretrained_networks.m
  * After download the pretrained networks such as Type I, and Type II, run the deom_fig5_fig7.m to reproduce the Fig. 5 and Fig. 7.

Trained network
===============
* Type I network for 'FBP-domain learing trainined a-1: [380, 1440] and a-2: [240, 380, 600, 1400] detectors dataset' is uploaded.
* Type II network for 'DBP-domain learing trainined b-1: [380, 1440] and b-2: [240, 380, 600, 1400] detectors dataset' is uploaded.

Test data
===============
* Iillustate the Fig. 5 for Framing U-Net via Deep Convolutional Framelets:Application to Sparse-view CT
* CT images from '2016 Low-Dose CT Grand Challenge' are uploaded to test.
Thanks Dr. Cynthia McCollough, the Mayo Clinic, the American Association of Physicists in Medicine(AAPM), and grand EB017095 and EB017185 from the National Institute of Biomedical Imaging and Bioengineering for providing the Low-Dose CT Grand Challenge dataset.
