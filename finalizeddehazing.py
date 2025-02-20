import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from numba import jit
from timeit import default_timer as timer
import pyfftw

class ImageDehazer:
    def __init__(self, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                 regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=False):
        self.airlightEstimation_windowSze = airlightEstimation_windowSze
        self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
        self.C0 = C0
        self.C1 = C1
        self.regularize_lambda = regularize_lambda
        self.sigma = sigma
        self.delta = delta
        self.showHazeTransmissionMap = showHazeTransmissionMap
        self._A = []
        self._Transmission = []
        self._WFun = []
        self.filter_bank = self.__LoadFilterBank()
        self.filter_otfs = None  # Precomputed OTFs

    def __timer(func):
        def wrapper(self, *args, **kwargs):
            start = timer()
            result = func(self, *args, **kwargs)
            end = timer()
            print(f"{func.__name__} executed in {end - start:.4f} seconds")
            return result
        return wrapper

    @__timer
    def __AirlightEstimation(self, HazeImg):
        kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
        if len(HazeImg.shape) == 3:
            with ThreadPoolExecutor() as executor:
                self._A = list(executor.map(lambda i: cv2.erode(HazeImg[:, :, i], kernel).max(), range(3)))
                print(f'__AirlightEstimation : Transmission type: {type(self._A)} {type(self._A[0])} {len(self._A)}')
        else:
            self._A = [cv2.erode(HazeImg, kernel).max()]

    @__timer
    def __BoundCon(self, HazeImg):
        if len(HazeImg.shape) == 3:
            t_b = np.maximum(
                (self._A[0] - HazeImg[:, :, 0].astype(np.float32)) / (self._A[0] - self.C0),
                (HazeImg[:, :, 0].astype(np.float32) - self._A[0]) / (self.C1 - self._A[0])
            )
            print(f'__BoundCon : A type: {type(self._A)}  {len(self._A)}')

            t_g = np.maximum(
                (self._A[1] - HazeImg[:, :, 1].astype(np.float32)) / (self._A[1] - self.C0),
                (HazeImg[:, :, 1].astype(np.float32) - self._A[1]) / (self.C1 - self._A[1])
            )
            t_r = np.maximum(
                (self._A[2] - HazeImg[:, :, 2].astype(np.float32)) / (self._A[2] - self.C0),
                (HazeImg[:, :, 2].astype(np.float32) - self._A[2]) / (self.C1 - self._A[2])
            )
            MaxVal = np.maximum(t_b, np.maximum(t_g, t_r))
            self._Transmission = np.minimum(MaxVal, 1)
            print(f'__BoundCon : Transmission type: {type(self._Transmission)}, {self._Transmission.shape}')
        else:
            self._Transmission = np.maximum(
                (self._A[0] - HazeImg.astype(np.float32)) / (self._A[0] - self.C0),
                (HazeImg.astype(np.float32) - self._A[0]) / (self.C1 - self._A[0])
            )
            self._Transmission = np.minimum(self._Transmission, 1)
        kernel = np.ones((self.boundaryConstraint_windowSze, self.boundaryConstraint_windowSze), float)
        self._Transmission = cv2.morphologyEx(self._Transmission, cv2.MORPH_CLOSE, kernel=kernel)

    @__timer
    @lru_cache(maxsize=1)
    def __LoadFilterBank(self):
        kernels = [
            [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
            [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
            [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
            [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
            [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
            [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
            [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
            [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        ]
        print(f'__LoadFilterBank : {type(kernels)} ')        

        print(f'__LoadFilterBank : {type([np.array(kernel, dtype=np.float32) / np.linalg.norm(kernel) for kernel in kernels])} ')        
       
        return [np.array(kernel, dtype=np.float32) / np.linalg.norm(kernel) for kernel in kernels]

    @__timer
    def __CalculateWeightingFunction(self, HazeImg, Filter):
        HazeImageDouble = HazeImg.astype(np.float32) / 255.0
        if len(HazeImg.shape) == 3:
            d_r = self.__circularConvFilt(HazeImageDouble[:, :, 2], Filter)
            d_g = self.__circularConvFilt(HazeImageDouble[:, :, 1], Filter)
            d_b = self.__circularConvFilt(HazeImageDouble[:, :, 0], Filter)
            return np.exp(-((d_r * 2) + (d_g * 2) + (d_b ** 2)) / (2 * self.sigma * self.sigma))
        else:
            d = self.__circularConvFilt(HazeImageDouble, Filter)
            return np.exp(-(3 * d ** 2) / (2 * self.sigma * self.sigma))

    @__timer
    @jit(nopython=True)
    def __circularConvFilt(self, Img, Filter):
        print("adhdvb")
        FilterHeight, FilterWidth = Filter.shape
        print(f' circularConvFilt : Filter type: {type(Filter)} ')

        assert FilterHeight == FilterWidth and FilterHeight % 2 == 1
        filterHalfSize = FilterHeight // 2
        PaddedImg = cv2.copyMakeBorder(Img, filterHalfSize, filterHalfSize,
                                     filterHalfSize, filterHalfSize,
                                     borderType=cv2.BORDER_WRAP)
        FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)

        return FilteredImg[filterHalfSize:-filterHalfSize, filterHalfSize:-filterHalfSize]

    @__timer
    def __CalTransmission(self, HazeImg):
        rows, cols = self._Transmission.shape
        
        
     #   print(f'__CalTransmission : filter_otfs type: {type(self.filter_otfs)} ')
     #    print(f'__CalTransmission : self.filter_bank: {type(self.filter_bank)} ')

        if self.filter_otfs is None:
            self.filter_otfs = [self.__psf2otf(kf, (rows, cols)) for kf in self.filter_bank]
       #     print(f'__CalTransmission : filter_otfs type: {type(self.filter_otfs)} ')

        DS = sum(abs(otf) ** 2 for otf in self.filter_otfs)
       # print(f'__CalTransmission : DS type: {type(DS)} ')

        tF = pyfftw.interfaces.numpy_fft.fft2(self._Transmission)
        beta = 1
        beta_max = 2 ** 4
        beta_rate = 2 * np.sqrt(2)
        while beta < beta_max:
            gamma = self.regularize_lambda / beta
         #   print(f'__CalTransmission : regularize_lambda type: {type(self.regularize_lambda)} ')
          #  print(f'__circularConvFilt : __circularConvFilt: {type(self.__circularConvFilt)} ')
            


           
            with ThreadPoolExecutor() as executor:
                DU_parts = list(executor.map(
                    lambda kf_wf: pyfftw.interfaces.numpy_fft.fft2(self.__circularConvFilt(
                        np.maximum(abs(self.__circularConvFilt(self._Transmission, kf_wf[0])) - kf_wf[1] / (len(self.filter_bank) * beta), 0) *
                        np.sign(self.__circularConvFilt(self._Transmission, kf_wf[0])),
                        cv2.flip(kf_wf[0], -1)
                    ))
                    
                    ,
                    zip(self.filter_bank, self._WFun)
                ))
             #   print(f'__CalTransmission : regularize_lambda type: {type(DU_parts)} ')
                DU = sum(DU_parts)
                self._Transmission = np.abs(pyfftw.interfaces.numpy_fft.ifft2((gamma * tF + DU) / (gamma + DS)))
                beta *= beta_rate

    @__timer
    def __psf2otf(self, psf, outSize):
        psfSize = psf.shape
        psf = np.pad(psf, [(0, outSize[0] - psfSize[0]), (0, outSize[1] - psfSize[1])], mode='constant')
        for i in range(len(psfSize)):
            psf = np.roll(psf, -int(psfSize[i] / 2), axis=i)
        otf = pyfftw.interfaces.numpy_fft.fft2(psf)
        nElem = np.prod(psfSize)
        otf[np.abs(otf) < nElem * np.finfo(float).eps] = 0
     #   print(f'__psf2otf : otf type: {type(otf)} {type(otf[0])}')
      #  print(f'__psf2otf : outSize type: {type(outSize)}')

        return otf

    @__timer
    def __removeHaze(self, HazeImg):
        epsilon = 0.0001
        Transmission = np.power(np.maximum(abs(self._Transmission), epsilon), self.delta)
        if len(HazeImg.shape) == 3:
            HazeCorrectedImage = np.zeros_like(HazeImg)
            for ch in range(3):
                temp = ((HazeImg[:, :, ch].astype(np.float32) - self._A[ch]) / Transmission) + self._A[ch]
                HazeCorrectedImage[:, :, ch] = np.clip(temp, 0, 255).astype(np.uint8)
           #     print(f'removehaze : A type: {type(self._A)} {len(self._A)}')
                

        else:
            temp = ((HazeImg.astype(np.float32) - self._A[0]) / Transmission) + self._A[0]
            HazeCorrectedImage = np.clip(temp, 0, 255).astype(np.uint8)
        return HazeCorrectedImage

    @__timer
    def deHaze(self, HazeImg):
        haze_img = HazeImg
        self.__AirlightEstimation(haze_img)
        self.__BoundCon(haze_img)
        self.__CalTransmission(haze_img)
        dehazed_small = self.__removeHaze(haze_img)
        dehazed_img = cv2.resize(dehazed_small, (HazeImg.shape[1], HazeImg.shape[0]), interpolation=cv2.INTER_LINEAR)
        return dehazed_img


def process_image(input_path, output_path=None):
    
    haze_img = cv2.imread(input_path)
    if haze_img is None:
        raise ValueError(f"Image at {input_path} could not be loaded.")
    
    dehazer = ImageDehazer()
    start = timer()
    dehazed_img = dehazer.deHaze(haze_img)
    end = timer()
   # print(f"Dehazing completed in {end - start:.4f} seconds")
    
    if output_path is None:
        output_path = input_path.replace(".jpg", "_dehazed.jpg").replace(".png", "_dehazed.png")
    cv2.imwrite("finalizeddehazing.jpg", dehazed_img)
    return output_path


process_image(r'C:\Users\Raymedis - IdeaPad\Downloads\handnew1.jpg')