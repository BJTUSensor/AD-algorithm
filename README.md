# AD-algorithm

Here, we have provided the code we used for Brillouin Distributed fiber sensors, the Brillouin gain spectra (BGS) data from our cooperative party, as well as the paper we published in Oct. 2020.

1. The Data: the BGS image was measured by Dr. Biwei Wang, the Hong Kong Polytechnic University. And it is open and offered out of Dr. Wang and his group's courtesy.
2. The core code: includes 3 parts, the 1st one is our AD denoising code, adapted by Mr. Kuo Luo (https://github.com/Lightningnemo) from a former repository by Mr. Michael Aye, https://github.com/michaelaye/pymars/blob/ca62a17c682f999c490cc0dbceb01433c385ced0/pymars/anisodiff2D.py. The 2nd is the BM3D code, and the last is a demo of how to use our code with the curve fitting part.
3. Abstract of the paper: Anisotropic diffusion (AD) is employed to enhance the signal-to-noise ratio (SNR) of Brillouin distributed optical fiber sensor. A Brillouin optical time-domain analyzer (BOTDA) with 99-km-long fiber under test is set up, where a section of 2.3 m and another 182 m section at the end of the fiber are heated for experimental verification. The SNR of experimental data sets with different sampling point numbers are enhanced to several improvement levels byADand three othermethods for comparison. Results showthat the 2.3msection and the temperature transition region of the 182msection are better preserved byADthan othermethods for the same SNR improvement. Objective criteria analysis shows that AD achieves the best measured spatial resolution and temperature accuracy among the four methods, the location of temperature transition can be detected more accurately for data with low SPNs afterADdenoising. In addition, the processing time ofADis 1/3 that of non-local means (NLM) and 6‰that of block-matching and 3D filtering (BM3D). The edge-preserving quality and fast processing speed allow the proposed algorithm to be a competitive denoising alternative for sensors based on Brillouin scattering.

We have provided _IEEE Journal of Lightwave Technology_ paper for better understanding. You are recommended to download the paper [1] directly from IEEE Journal of Lightwave Technology due to the copy right issues.

[1]. Kuo Luo, Biwei Wang, Nan Guo, Kuanglu Yu, et al, Enhancing SNR by Anisotropic Diffusion for Brillouin Distributed Optical Fiber Sensors, IEEE Journal of Lightwave Technology, 38, 5844-5852 (2020).

First Online Date: 22:00 Beijing Time, Jan. 12th, 2022
