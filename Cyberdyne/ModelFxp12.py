import torch
import numpy as np
from pathlib import Path
from operator import itemgetter

class Conv:
    
    # Log2 Quantization constructor
    def __init__(self, torchLayer, weight_shift, bias_shift):
        
        self.weight_shift = weight_shift
        self.weight_offset = np.power(2, weight_shift)
        self.bias_shift = bias_shift
        self.bias_offset = np.power(2, bias_shift)
        self.pad = torchLayer.padding
        self.stride = torchLayer.stride
        
        if self.pad[0] > 1:
            print('WARNING: Hardware supports only padding = 0 or padding = 1')
        

        # Quantization in action
        w_fxp      = (torchLayer.weight.detach().numpy() * self.weight_offset)  # Convert to fixed point
        w_fxp      = np.where(w_fxp == 0, 0.5, w_fxp) # 0.5 = 2^-1
        w_fxp_sign = np.sign(w_fxp)                                # Save the weights sign 
        w_fxp_log2 = np.log2(np.abs(w_fxp)).round().astype('int8') # Obtain the shift amount used in multiplication
        # w_fxp_log2 < 0 means the weight is between 0 and 1 or 0 and -1.
        # Such weight is marked as -1 to result 0 when multiplying the feature.
        # In software a the abs(-1) shift is huge and results 0.
        # In hardware, when generating weights memory images, the shift amount -1 is substituted by 0
        #     VHDL: conv_mult_s(f) <=   (others=>'0') when SIGNED(filter_weight_s(f)) = 0
        w_fxp_log2[w_fxp_log2 < 0] = -1
        
        b_fxp      = (torchLayer.bias.detach().numpy() * self.bias_offset).astype('int16')         # Convert to fixed point integer

        self.w = w_fxp_log2         # Unsigned weigths
        self.w_sign = w_fxp_sign    # Weights sign
        self.b = b_fxp              # Signed biases    
        
        self.smallest = 0
        self.highest = 0

    def Predict(self, state):

        N, C, H, W   = state.shape
        F, _, HH, WW = self.w.shape # fmap1: (8, 1, 8, 8)
                
        # dimensions of the output
        H_ = int(1 + (H + 2 * self.pad[0] - HH) / self.stride[0])
        W_ = int(1 + (W + 2 * self.pad[1] - WW) / self.stride[1])

        # output definition
        out = np.zeros((N, F, H_, W_)).astype('int32')

        # 0-padding just on two dimentions of x
        z = np.pad(state, ((0,), (0,), (self.pad[0],), (self.pad[1],)), 'constant')
        
        for f in range(F):
            for i in range(H_):
                for j in range(W_):

                    h = self.stride[0] * i
                    w = self.stride[1] * j
        
                    mult = np.left_shift(z[:, :, h:h + HH, w:w + WW].astype('int32'),  self.w[f][:].astype('uint8')) # Multiplications using shift.
                    mult_rsh = np.right_shift(mult, self.weight_shift)    # Remove extra bits from the multiplications result
                    mult_rsh_sign = np.multiply(mult_rsh, self.w_sign[f])   # Set the values signs
                    sum = np.sum(mult_rsh_sign, axis=(1,2,3))                  
                    out[:, f, i, j] = out[:, f, i, j] + sum    # Accumulate
 
            out[:, f] = out[:, f] + self.b[f]  # Add bias  
    
        # Used to adjust the minimum accumulator width in hardware for convolutional layers          
        if out.min() < self.smallest:
            self.smallest = out.min()
            
        if out.max() > self.highest:
            self.highest = out.max()
        
        return out
     
       

    def __call__(self, state):
        return self.Predict(state)
    
class Linear:
    
    # Log2 Quantization constructor
    def __init__(self, torchLayer, weight_shift, bias_shift):
        
        self.weight_shift = weight_shift
        self.weight_offset = np.power(2, weight_shift)
        self.bias_shift = bias_shift
        self.bias_offset = np.power(2, bias_shift)

        # Quantization in action
        w_fxp      = torchLayer.weight.detach().numpy() * self.weight_offset                     # Convert to fixed point
        w_fxp      = np.where(w_fxp == 0, 0.5, w_fxp) # 0.5 = 2^-1
        w_fxp_sign = np.sign(w_fxp)                                 # Save the weights sign 
        w_fxp_log2 = np.log2(np.abs(w_fxp)).round().astype('int8') # Obtain the shift amount used in multiplication
        # w_fxp_log2 < 0 means the weight is between 0 and 1 or 0 and -1.
        # Such weight is marked as -1 to result 0 when multiplying the feature.
        # In software a the abs(-1) shift is huge and results 0.
        # In hardware, when generating weights memory images, the shift amount -1 is substituted by 0
        #     VHDL: neuron_mult_s(i) <= (others=>'0') when SIGNED(neuron_weight_s(i)) = 0
        w_fxp_log2[w_fxp_log2 < 0] = -1  
        
        b_fxp = ((torchLayer.bias.detach().numpy()) * self.bias_offset).astype('int16')         # Convert to fixed point integer

        self.w = w_fxp_log2         # Unsigned weigths
        self.w_sign = w_fxp_sign    # Weights sign
        self.b = b_fxp              # Signed biases
        
        self.smallest = 0
        self.highest = 0
        
        
    def Predict(self, state):        
        
        # output definition
        out = np.zeros((state.shape[0], self.w.shape[0])).astype('int32')
        
        for i in range(state.shape[0]):
                
            mult = np.left_shift(state[i].astype('int32'), self.w.astype('uint8'))   # Multiplications using shift.
            mult_rsh = np.right_shift(mult, self.weight_shift)    # Remove extra bits from the multiplications result
            mult_rsh_sign = np.multiply(mult_rsh, self.w_sign)  # Set the values signs
            sum = np.sum(mult_rsh_sign, axis = 1)   # Sumation
            out[i] = (sum + self.b).astype('int32') # Add bias
            
        # Used to adjust the minimum accumulator width in hardware for full connected layers
        if out.min() < self.smallest:
            self.smallest = out.min()
            
        if out.max() > self.highest:
            self.highest = out.max()
          
        return out
    
    
    def __call__(self, state):
        return self.Predict(state)

# Considers only square kernels and stride equals to kernel
class MaxPool2d:
    
    def __init__(self, kernel_order):
        self.k = kernel_order
        
    def Forward(self, fm): # Input: fm(n, c, h, w)
        out = np.zeros((fm.shape[0], fm.shape[1], (fm.shape[2] - self.k) // self.k + 1, (fm.shape[3] - self.k) // self.k + 1)).astype('int32')
        
        for zh in range(0, fm.shape[2] - self.k + 1, self.k):
            for zw in range(0, fm.shape[3] - self.k + 1, self.k):
                out[:, :, (zh // self.k), (zw // self.k)] = np.max(fm[:, :, zh:zh + self.k, zw:zw + self.k], axis=(2,3))
        
        
        return out
    
    def __call__(self, fm):
        return self.Forward(fm)
        
class ReLU:
    
    def __init__(self):
        self.reluMax = 0
        
    def Forward(self, x):
        
        y = np.maximum(x, 0)
        
        if y.max() > self.reluMax:
            self.reluMax = y.max()
        
        return y
    
    def __call__(self, x):
        return self.Forward(x)



################# TEST LAYERS ####################################
### Placed only after ReLU ###
class ActivationsToPowerOfTwoValues:
        
    def Forward(self, act):
        act = np.where(act == 0, 1<<32, act)
        exp = np.round(np.log2(act)).astype('uint32')

        p2value =  (1 << exp).astype('uint32')
        
        return p2value
    
    def __call__(self, act):
        return self.Forward(act)
    
class SqueezeActivations:
        
    def __init__(self, nbits):
        self.nbits = nbits
    
    def Forward(self, act):
                
        actRedux = np.right_shift(act.astype('uint32'), self.nbits)
                
        actRestored = np.left_shift(actRedux, self.nbits)
                
        return actRestored
    
    def __call__(self, act):
        return self.Forward(act)
    
class SqueezeActivationsRound:
        
    def __init__(self, nbits):
        self.nbits = nbits
    
    def Forward(self, act):
                
        actRedux = (act / np.power(2, self.nbits)).round().astype('uint32') 
                
        actRestored = np.left_shift(actRedux, self.nbits)
                
        return actRestored
    
    def __call__(self, act):
        return self.Forward(act)
        
class SqueezeActivationsRoundOr:
        
    def __init__(self, nbits):
        self.nbits = nbits
    
    def Forward(self, act):

        b = np.bitwise_and(act.astype('uint32'), (1<<(self.nbits-1))) 

        c = np.bitwise_or(act.astype('uint32'), b<<1)

        actRestored = (c>>self.nbits) << self.nbits
        
        return actRestored
    
    def __call__(self, act):
        return self.Forward(act)
######### END OF TEST LAYERS ####################################




OutputShape = lambda ih, iw, kh, kw, ph, pw, sh, sw: ((ih - kh + 2*ph + sh) // sh, (iw - kw + 2*pw + sw) // sw)

  
    

class ModelFxp():
    
    ### SUPORTED ONLY WEIGHTS_PER_LINE  = 8 ###
    WEIGHTS_PER_LINE = 8 ######################             
    ###########################################
    
    ### MAXIMUM SIZE = 4 BITS ####
    WEIGHT_WIDTH = 4 #############
    #############################
    
    WEIGHTS_MEM_DATA_WIDTH = WEIGHTS_PER_LINE * WEIGHT_WIDTH
  
    SIGNS_MEM_DATA_WIDTH = WEIGHTS_PER_LINE
    
    def __init__(self, layers):
        
        self.layers = layers
        self.bias_offset = layers[0].bias_offset
             
        self.ConvLayers = []
        self.FullLayers = []
        for (l, i) in zip(self.layers, range(len(self.layers))):
            print(f'layers[{i}]: {type(l).__name__}')
            
            if isinstance(l, Conv):
                self.ConvLayers.append(l)
                
                # Verify if the min/max weight was not exceed
                if 0 in l.w:
                    print(f'MIN WEIGHT(exp) EXCEED ON CONV at layers[{i}]. Increase shift.')
                    
                if l.w.max() > np.power(2, ModelFxp.WEIGHT_WIDTH) - 1:
                    print(f'MAX WEIGHT(exp) EXCEED ON CONV at layers[{i}]. Decrease shift.')
            
            elif isinstance(l, Linear):
                self.FullLayers.append(l)
                
                # Verify if the min/max weight was not exceed
                if 0 in l.w:
                    print(f'MIN WEIGHT(exp) EXCEED ON FULL at layers[{i}]. Increase shift.')

                if l.w.max() > np.power(2, ModelFxp.WEIGHT_WIDTH) - 1:
                    print(f'MAX WEIGHT(exp) EXCEED ON FULL at layers[{i}]. Decrease shift.')

    
    def Forward(self, x):
        a = (x * self.bias_offset).numpy().astype('int16')
       
        for l in self.layers:
            if isinstance(l, Linear):
                a = a.reshape(a.shape[0], -1)
            
            a = l(a)
           
        return a
    
    def AccumulatorsWidth(self):
        for i in range(len(self.ConvLayers)):
            smallest = self.ConvLayers[i].smallest
            highest = self.ConvLayers[i].highest
            
            if np.abs(smallest) > np.abs(highest):
                width = np.ceil(np.log2(np.abs(np.abs(smallest)))) + 1
            else:
                width = np.ceil(np.log2(np.abs(np.abs(highest)))) + 1
            
            print(f'ConvLayers[{i}]: min = {smallest}, max = {highest} MIN_WIDTH = {width}')
            
        for i in range(len(self.FullLayers)):
            smallest = self.FullLayers[i].smallest
            highest = self.FullLayers[i].highest
            
            if np.abs(smallest) > np.abs(highest):
                width = np.ceil(np.log2(np.abs(np.abs(smallest)))) + 1
            else:
                width = np.ceil(np.log2(np.abs(np.abs(highest)))) + 1
            
            print(f'FullLayers[{i}]: min = {smallest}, max = {highest} MIN_WIDTH = {width}')
    
    def WeightsWidth(self):
        for i in range(len(self.ConvLayers)):
            print(f'ConvLayers[{i}] min width: {np.ceil(np.log2(np.abs(self.ConvLayers[i].w).max()))}')
            
        for i in range(len(self.FullLayers)):
            print(f'FullLayers[{i}] min width: {np.ceil(np.log2(np.abs(self.FullLayers[i].w).max()))}')
    
    def __call__(self, x):
        return self.Forward(x)  
    
    def MemoryImageConv(self, directory, mif = False):
                
        mem_depth = 0
        for cl in self.ConvLayers:
            mem_depth += np.ceil(((cl.w.shape[1] * cl.w.shape[2] * cl.w.shape[3]) * ModelFxp.WEIGHT_WIDTH) / ModelFxp.WEIGHTS_MEM_DATA_WIDTH) + 1 # 1 = bias 
                
        FILTER_MEM_DEPTH = np.power(2, np.ceil(np.log2(mem_depth))).astype('int32')
                
        self.filter_start_addr = np.zeros(len(self.ConvLayers)).astype('uint32')
        
        sortedConvLayers = [] # Initially not sorted
       
        # Group conv layers index and filters amount
        for (convLayer, i) in zip(self.ConvLayers, range(len(self.ConvLayers))):
            sortedConvLayers.append((i, convLayer.w.shape[0]))
        
        print('*** CONVOLUTIONAL LAYERS ***\n')
        
        # Sort conv layers by filters amount
        sortedConvLayers.sort(key=itemgetter(1), reverse=True)
        print(f'sortedConvLayers (index, filters): {sortedConvLayers}')
        self.maxFilters = sortedConvLayers[0][1]
        print(f'self.maxFilters: {self.maxFilters}, convLayer[{sortedConvLayers[0][0]}]\n')
        
        
        filterWeights = []
        filterSigns = []
        weightsPadding = np.ceil(ModelFxp.WEIGHTS_MEM_DATA_WIDTH / 4).astype('uint8')
        signsPadding = np.ceil(ModelFxp.WEIGHTS_PER_LINE / 4).astype('uint8')
        
        ############################################
        ### Generates filters memory image files ###
        ############################################
        for filter in range(self.maxFilters):
            if mif:
                Path(f'{directory}/mif').mkdir(exist_ok=True, parents=True)
                
                filterWeightsFile = open(f'{directory}/mif/FilterWeights_{filter:03}.mif', "w+")
                # mif header
                filterWeightsFile.write(f'DEPTH = {FILTER_MEM_DEPTH};\n')
                filterWeightsFile.write(f'WIDTH = {ModelFxp.WEIGHTS_MEM_DATA_WIDTH};\n')
                filterWeightsFile.write(f'ADDRESS_RADIX = DEC;\n')
                filterWeightsFile.write(f'DATA_RADIX = HEX;\n')
                filterWeightsFile.write(f'CONTENT\n')
                filterWeightsFile.write(f'BEGIN\n\n')
                    
                filterSignsFile = open(f'{directory}/mif/FilterSigns_{filter:03}.mif', "w+")
                # mif header
                filterSignsFile.write(f'DEPTH = {FILTER_MEM_DEPTH};\n')
                filterSignsFile.write(f'WIDTH = {ModelFxp.SIGNS_MEM_DATA_WIDTH};\n')
                filterSignsFile.write(f'ADDRESS_RADIX = DEC;\n')
                filterSignsFile.write(f'DATA_RADIX = HEX;\n')
                filterSignsFile.write(f'CONTENT\n')
                filterSignsFile.write(f'BEGIN\n\n')
                nl = ';\n'
            else:
                Path(f'{directory}/raw').mkdir(exist_ok=True, parents=True)
                
                filterWeightsFile = open(f'{directory}/raw/FilterWeights_{filter:03}.txt', "w+")
                filterSignsFile = open(f'{directory}/raw/FilterSigns_{filter:03}.txt', "w+")
                nl = '\n'
            
            addrCountWeights = 0
            addrCountSigns = 0
            filterWeights.append(filterWeightsFile.name[filterWeightsFile.name.find("raw/"):])
            filterSigns.append(filterSignsFile.name[filterSignsFile.name.find("raw/"):])
        
            for cl in sortedConvLayers:  
                
                # Verifies if the current layer has at least 'filter' filters
                if filter < self.ConvLayers[cl[0]].w.shape[0]:               
                    i = 0
                    shamt = ModelFxp.WEIGHTS_MEM_DATA_WIDTH - ModelFxp.WEIGHT_WIDTH
                    weightsWord = 0
                    if filter == 0: 
                        print(f'FILTER_WEIGHTS[{addrCountWeights}]: begin of convolution layer {cl[0]} filters')
                        self.filter_start_addr[cl[0]] = addrCountWeights
                    
                    
                    ### Write the weights to the file considering 32 bits memory word        
                    for w in self.ConvLayers[cl[0]].w[filter].flatten():
                        if w == -1:
                            w = 0  # This assignment changes data type from 'numpy.int8' to 'int'
                        
                        if mif and i == 0:
                            filterWeightsFile.write(f'{addrCountWeights}:')
                        
                        if i == ModelFxp.WEIGHTS_PER_LINE - 1:
                            weightsWord |= (np.uint32(w) << shamt)
                            filterWeightsFile.write('{0:0{1}X}'.format(weightsWord, weightsPadding))
                            filterWeightsFile.write(nl)
                            shamt = ModelFxp.WEIGHTS_MEM_DATA_WIDTH - ModelFxp.WEIGHT_WIDTH
                            weightsWord = 0 
                            addrCountWeights += 1
                            i = 0                           
                        else:                            
                            weightsWord |= (np.uint32(w) << shamt)
                            shamt -= ModelFxp.WEIGHT_WIDTH
                            i += 1
                            
                    # Filler characters to fulfill a line
                    if i > 0:
                        filterWeightsFile.write('{0:0{1}X}'.format(weightsWord, weightsPadding))
                        filterWeightsFile.write(nl)
                        addrCountWeights += 1                        
                        
                    if filter == 0: 
                        print(f'FILTER_WEIGHTS[{addrCountWeights}]: convolution layer {cl[0]} filter bias')
                    
                    # Insert bias after kernel weights    
                    bias = self.ConvLayers[cl[0]].b[filter].astype('uint32')
                    
                    # Format bias according to the WEIGHTS_MEM_DATA_WIDTH
                    bias = (bias << (32 - ModelFxp.WEIGHTS_MEM_DATA_WIDTH)).astype('uint32')
                    bias >>= (32 - ModelFxp.WEIGHTS_MEM_DATA_WIDTH)
                    
                    if mif:
                        filterWeightsFile.write(f'{addrCountWeights}:')
                    
                    filterWeightsFile.write('{0:0{1}X}'.format(bias, weightsPadding))
                    filterWeightsFile.write(nl)
                    addrCountWeights += 1
                    
                    
                    
                    ### Write the weight signs to the file considering 8 bits memory word                    
                    ws = self.ConvLayers[cl[0]].w_sign[filter].flatten()
                    # ws is float64: minus = -1 and plus = 1
                    ws[ws == 1] = 0     # Converts float plus to 2 complement plus 
                    ws[ws == -1] = 1    # Converts float minus to 2 complement minus
                    ws = ws.astype('uint8') # In order to print as hex
                    i = 0
                    ac = 0
                    for s in ws.flatten():                            
                        if mif and i == 0:
                            filterSignsFile.write(f'{addrCountSigns}:')
                                
                        ac += s * (2**(ModelFxp.SIGNS_MEM_DATA_WIDTH - 1 - i))    # Converts ModelFxp.SIGNS_MEM_DATA_WIDTH signs to an hex number
                        if i == ModelFxp.SIGNS_MEM_DATA_WIDTH - 1:
                            filterSignsFile.write('{0:0{1}X}'.format(ac, signsPadding))
                            filterSignsFile.write(nl)
                            addrCountSigns += 1
                            ac = 0
                            i = 0
                        else:
                            i += 1
            
                    if i > 0 and i < ModelFxp.SIGNS_MEM_DATA_WIDTH:
                        filterSignsFile.write('{0:0{1}X}'.format(ac, signsPadding))
                        filterSignsFile.write(nl)
                        addrCountSigns += 1
                        
                    # Bias sign. Used only to keep the same format as the weights memory
                    if mif:
                        filterSignsFile.write(f'{addrCountSigns}:')
                            
                    filterSignsFile.write('{0:0{1}X}'.format(0, signsPadding))
                    filterSignsFile.write(nl)
                    addrCountSigns += 1
                      
            # Filler lines
            for _ in range(FILTER_MEM_DEPTH - addrCountWeights):
                if mif:
                    filterWeightsFile.write(f'{addrCountWeights}:')
                    filterSignsFile.write(f'{addrCountWeights}:')
                    addrCountWeights += 1
                
                filterWeightsFile.write('{0:0{1}X}'.format(0, weightsPadding))
                filterWeightsFile.write(nl)
                filterSignsFile.write('{0:0{1}X}'.format(0, signsPadding))
                filterSignsFile.write(nl) 
                    
            if mif:
                filterWeightsFile.write(f'END;\n')
                filterSignsFile.write(f'END;\n')
            
            if filter == 0:
                print(f'\nData width: {ModelFxp.WEIGHTS_MEM_DATA_WIDTH}')
                filters_addr_width = np.ceil(np.log2(addrCountWeights)).astype('int32')
                print(f'Required filter memories depth: {addrCountWeights} ({addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits). BRAM depth: {np.power(2, filters_addr_width)} ({np.power(2, filters_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits) (FILTERS_ADDR_WIDTH = {filters_addr_width})')
                print(f'Total required memory: {self.maxFilters} x {addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} = {self.maxFilters * addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits. Total BRAM: {self.maxFilters} x {np.power(2, filters_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} = {self.maxFilters * np.power(2, filters_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits')
            
            filterWeightsFile.close()
            filterSignsFile.close() 
                
        return filterWeights, filterSigns, filters_addr_width 

    def MemoryImageFC(self, directory, mif = False):
                    
        ############################################
        ### Generates neurons memory image files ###
        ############################################
        mem_depth = 0
        for fl in self.FullLayers:
            mem_depth += np.ceil((fl.w.shape[1] * ModelFxp.WEIGHT_WIDTH) / ModelFxp.WEIGHTS_MEM_DATA_WIDTH) + 1  # 1 = 32-bits bias 
        
        NEURON_MEM_DEPTH = np.power(2, np.ceil(np.log2(mem_depth))).astype('int32')
        
        self.weight_start_addr = np.zeros(len(self.FullLayers)).astype('uint32')
        
        self.sortedFullLayers = [] # Initialy not sorted
       
        # Group full connected layers index and filters amount
        for (fcLayer, i) in zip(self.FullLayers, range(len(self.FullLayers))):
            self.sortedFullLayers.append((i, fcLayer.w.shape[0]))
        
        print('\n\n\n*** FULL CONNECTED LAYERS ***\n')
        
        # Sort full connected layers by neurons amount
        self.sortedFullLayers.sort(key=itemgetter(1), reverse=True)
        print(f'sortedFullLayers (index, neurons): {self.sortedFullLayers}')
        self.maxNeurons = self.sortedFullLayers[0][1]
        print(f'self.maxNeurons: {self.maxNeurons}, FullLayer[{self.sortedFullLayers[0][0]}]\n')
        
        neuronWeights = []
        neuronSigns = []
        weightsPadding = np.ceil(ModelFxp.WEIGHTS_MEM_DATA_WIDTH / 4).astype('uint8')
        signsPadding = np.ceil(ModelFxp.WEIGHTS_PER_LINE / 4).astype('uint8')
        
        for neuron in range(self.maxNeurons):
            if mif:
                Path(f'{directory}/mif').mkdir(exist_ok=True, parents=True)
                neuronWeightsFile = open(f'{directory}/mif/NeuronWeights_{neuron:03}.mif', "w+")
                # mif header
                neuronWeightsFile.write(f'DEPTH = {NEURON_MEM_DEPTH};\n')
                neuronWeightsFile.write(f'WIDTH = {ModelFxp.WEIGHTS_MEM_DATA_WIDTH};\n')
                neuronWeightsFile.write(f'ADDRESS_RADIX = DEC;\n')
                neuronWeightsFile.write(f'DATA_RADIX = HEX;\n')
                neuronWeightsFile.write(f'CONTENT\n')
                neuronWeightsFile.write(f'BEGIN\n\n')

                neuronSignsFile = open(f'{directory}/mif/NeuronSigns_{neuron:03}.mif', "w+")
                # mif header
                neuronSignsFile.write(f'DEPTH = {NEURON_MEM_DEPTH};\n')
                neuronSignsFile.write(f'WIDTH = {ModelFxp.SIGNS_MEM_DATA_WIDTH};\n')
                neuronSignsFile.write(f'ADDRESS_RADIX = DEC;\n')
                neuronSignsFile.write(f'DATA_RADIX = HEX;\n')
                neuronSignsFile.write(f'CONTENT\n')
                neuronSignsFile.write(f'BEGIN\n\n')
                nl = ';\n'
            else:
                Path(f'{directory}/raw').mkdir(exist_ok=True, parents=True)
                neuronWeightsFile = open(f'{directory}/raw/NeuronWeights_{neuron:03}.txt', "w+")
                neuronSignsFile = open(f'{directory}/raw/NeuronSigns_{neuron:03}.txt', "w+")
                nl = '\n'
                
            addrCountWeights = 0
            addrCountSigns = 0
            neuronWeights.append(neuronWeightsFile.name[neuronWeightsFile.name.find("raw/"):])
            neuronSigns.append(neuronSignsFile.name[neuronSignsFile.name.find("raw/"):])
            
            for fc in self.sortedFullLayers:
                
                # Verifies if the current layer has at least 'neuron' neurons
                if neuron < self.FullLayers[fc[0]].w.shape[0]:
                    i = 0
                    shamt = ModelFxp.WEIGHTS_MEM_DATA_WIDTH - ModelFxp.WEIGHT_WIDTH
                    weightsWord = 0
                    if neuron == 0: 
                        print(f'NEURON_WEIGHTS[{addrCountWeights}]: begin of neuron {fc[0]} weights')
                        self.weight_start_addr[fc[0]] = addrCountWeights
                        
                    ### Write the weights to the file considering 32 bits memory word 
                    for w in self.FullLayers[fc[0]].w[neuron]:
                        if w == -1:
                            w = 0  # This assignement changes type from 'numpy.int8' to 'int'

                        if mif and i == 0:
                            neuronWeightsFile.write(f'{addrCountWeights}:')

                        if i == ModelFxp.WEIGHTS_PER_LINE - 1:
                            weightsWord |= (np.uint32(w) << shamt)
                            neuronWeightsFile.write('{0:0{1}X}'.format(weightsWord, weightsPadding))
                            neuronWeightsFile.write(nl)
                            shamt = ModelFxp.WEIGHTS_MEM_DATA_WIDTH - ModelFxp.WEIGHT_WIDTH
                            weightsWord = 0 
                            addrCountWeights += 1
                            i = 0                           
                        else:
                            weightsWord |= (np.uint32(w) << shamt)
                            shamt -= ModelFxp.WEIGHT_WIDTH
                            i += 1

                    # Filler characters to fulfill a line
                    if i > 0:
                        neuronWeightsFile.write('{0:0{1}X}'.format(weightsWord, weightsPadding))
                        neuronWeightsFile.write(nl)
                        addrCountWeights += 1
                        

                    if neuron == 0: 
                        print(f'NEURON_WEIGHTS[{addrCountWeights}]: neuron {fc[0]} bias')

                    # Insert bias after neuron weights    
                    bias = self.FullLayers[fc[0]].b[neuron].astype('uint32')
                    
                    # Format bias according to the WEIGHTS_MEM_DATA_WIDTH
                    bias = (bias << (32 - ModelFxp.WEIGHTS_MEM_DATA_WIDTH)).astype('uint32')
                    bias >>= (32 - ModelFxp.WEIGHTS_MEM_DATA_WIDTH)
                    
                    if mif:
                        neuronWeightsFile.write(f'{addrCountWeights}:')
                        
                    neuronWeightsFile.write('{0:0{1}X}'.format(bias, weightsPadding))
                    neuronWeightsFile.write(nl)
                    addrCountWeights += 1

                    ### Write the weight signs to the file considering 8 bits memory word                    
                    neuron_sign = self.FullLayers[fc[0]].w_sign[neuron].copy().flatten()
                    # neuron_sign is float64: minus = -1 and plus = 1
                    neuron_sign[neuron_sign == 1] = 0     # Converts float plus to 2 complement plus 
                    neuron_sign[neuron_sign == -1] = 1    # Converts float minus to 2 complement minus
                    neuron_sign = neuron_sign.astype('uint8') # In order to print as hex
                    i = 0
                    ac = 0
                    for s in neuron_sign:                            
                        if mif and i == 0:
                            neuronSignsFile.write(f'{addrCountSigns}:')

                        ac += s * (2**(ModelFxp.SIGNS_MEM_DATA_WIDTH - 1 - i))    # Converts ModelFxp.SIGNS_MEM_DATA_WIDTH signs to an hex number
                        if i == ModelFxp.SIGNS_MEM_DATA_WIDTH - 1:
                            neuronSignsFile.write('{0:0{1}X}'.format(ac, signsPadding))
                            neuronSignsFile.write(nl)
                            addrCountSigns += 1
                            ac = 0
                            i = 0
                        else:
                            i += 1

                    if i > 0 and i < ModelFxp.SIGNS_MEM_DATA_WIDTH:
                        neuronSignsFile.write('{0:0{1}X}'.format(ac, signsPadding))
                        neuronSignsFile.write(nl)
                        addrCountSigns += 1

                    # Bias sign. Used only to keep the same format as the weights memory
                    if mif:
                        neuronSignsFile.write(f'{addrCountSigns}:')

                    neuronSignsFile.write('{0:0{1}X}'.format(0, signsPadding))
                    neuronSignsFile.write(nl)
                    addrCountSigns += 1                
            
                
                
            # Filler lines
            for _ in range(NEURON_MEM_DEPTH - addrCountWeights):
                if mif:
                    neuronWeightsFile.write(f'{addrCountWeights}:')
                    neuronSignsFile.write(f'{addrCountWeights}:')
                    addrCountWeights += 1

                neuronWeightsFile.write('{0:0{1}X}'.format(0, weightsPadding))
                neuronWeightsFile.write(nl)
                neuronSignsFile.write('{0:0{1}X}'.format(0, signsPadding))
                neuronSignsFile.write(nl) 

            if mif:
                neuronWeightsFile.write(f'END;\n')
                neuronSignsFile.write(f'END;\n')
                
            if neuron == 0:
                neurons_addr_width = np.ceil(np.log2(addrCountWeights)).astype('int32')
                print(f'\nData width: {ModelFxp.WEIGHTS_MEM_DATA_WIDTH}')
                print(f'Required neuron memories depth: {addrCountWeights} ({addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits). BRAM depth: {np.power(2, neurons_addr_width)} ({np.power(2, neurons_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits) (NEURONS_ADDR_WIDTH = {neurons_addr_width})')
                print(f'Total required memory: {self.maxNeurons} * {addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} = {self.maxNeurons * addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits. Total BRAM: {self.maxNeurons} * {np.power(2, neurons_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} = {self.maxNeurons * np.power(2, neurons_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits')
            
            neuronWeightsFile.close()
            neuronSignsFile.close()
                
        return neuronWeights, neuronSigns, neurons_addr_width  

    def MemoryImageSerialFC(self, directory, mif = False):
                    
        ############################################
        ### Generates neurons memory image files ###
        ############################################
        
        print('\n\n\n*** FULL CONNECTED LAYERS ***\n')
        
        mem_depth_acc = 0
        
        for l, fl in enumerate(self.FullLayers):            
            mem_depth_acc += np.ceil((fl.w.shape[1] * ModelFxp.WEIGHT_WIDTH) / ModelFxp.WEIGHTS_MEM_DATA_WIDTH) * fl.w.shape[0] + len(fl.b) 
            FULL_LAYER_MEM_DEPTH = np.power(2, np.ceil(np.log2(mem_depth_acc))).astype('uint32')
            full_layer_mem_addr_width = np.log2(FULL_LAYER_MEM_DEPTH).astype('uint32')

        if mif:
            Path(f'{directory}/mif').mkdir(exist_ok=True, parents=True)
            weightsFile = open(f'{directory}/mif/FullLayer_weights.mif', "w+")
            # mif header
            weightsFile.write(f'DEPTH = {FULL_LAYER_MEM_DEPTH};\n')
            weightsFile.write(f'WIDTH = {ModelFxp.WEIGHTS_MEM_DATA_WIDTH};\n')
            weightsFile.write(f'ADDRESS_RADIX = DEC;\n')
            weightsFile.write(f'DATA_RADIX = HEX;\n')
            weightsFile.write(f'CONTENT\n')
            weightsFile.write(f'BEGIN\n\n')
            signsFile = open(f'{directory}/mif/FullLayer_signs.mif', "w+")
            # mif header
            signsFile.write(f'DEPTH = {FULL_LAYER_MEM_DEPTH};\n')
            signsFile.write(f'WIDTH = {ModelFxp.SIGNS_MEM_DATA_WIDTH};\n')
            signsFile.write(f'ADDRESS_RADIX = DEC;\n')
            signsFile.write(f'DATA_RADIX = HEX;\n')
            signsFile.write(f'CONTENT\n')
            signsFile.write(f'BEGIN\n\n')
            nl = ';\n'
        else:
            Path(f'{directory}/raw').mkdir(exist_ok=True, parents=True)
            weightsFile = open(f'{directory}/raw/FullLayer_weights.txt', "w+")
            signsFile = open(f'{directory}/raw/FullLayer_signs.txt', "w+")
            nl = '\n'
        
        self.maxNeurons = 0        
        
        addrCountWeights = 0
        addrCountSigns = 0
        
        for l, fl in enumerate(self.FullLayers):
            
            if fl.w.shape[0] > self.maxNeurons:
                self.maxNeurons = fl.w.shape[0]                
            
            weightsPadding = np.ceil(ModelFxp.WEIGHTS_MEM_DATA_WIDTH / 4).astype('uint8')
            signsPadding = np.ceil(ModelFxp.WEIGHTS_PER_LINE / 4).astype('uint8')
            
            ### Write the weights of each leayer neuron ### 
            for n in range(fl.w.shape[0]):
            
                i = 0  
                shamt = ModelFxp.WEIGHTS_MEM_DATA_WIDTH - ModelFxp.WEIGHT_WIDTH
                weightsWord = 0
                    
                ### Write the weights to the file considering 32 bits memory word ###
                for w in fl.w[n]:
                    if w == -1:
                        w = 0   # This assignment changes the type from 'numpy.int8' to 'int'

                    if mif and i == 0:
                        weightsFile.write(f'{addrCountWeights}:')

                    #weightsFile.write(f'{w & 0xf:X}')
                    if i == ModelFxp.WEIGHTS_PER_LINE - 1:
                        weightsWord |= (np.uint32(w) << shamt)
                        weightsFile.write('{0:0{1}X}'.format(weightsWord, weightsPadding))
                        weightsFile.write(nl)
                        shamt = ModelFxp.WEIGHTS_MEM_DATA_WIDTH - ModelFxp.WEIGHT_WIDTH
                        weightsWord = 0
                        addrCountWeights += 1
                        i = 0                           
                    else:
                        weightsWord |= (np.uint32(w) << shamt)
                        shamt -= ModelFxp.WEIGHT_WIDTH
                        i += 1

                # Filler characters to fulfill a line
                if i > 0:
                    #while(i < ModelFxp.WEIGHTS_PER_LINE):
                    #    weightsFile.write('0')
                    #    i += 1
                    weightsFile.write('{0:0{1}X}'.format(weightsWord, weightsPadding))
                    weightsFile.write(nl)
                    addrCountWeights += 1

                # Insert bias after neuron weights    
                bias = fl.b[n].astype('uint32')
                
                # Format bias according to the WEIGHTS_MEM_DATA_WIDTH
                bias = (bias << (32 - ModelFxp.WEIGHTS_MEM_DATA_WIDTH)).astype('uint32')
                bias >>= (32 - ModelFxp.WEIGHTS_MEM_DATA_WIDTH)
                
                if mif:
                    weightsFile.write(f'{addrCountWeights}:')

                weightsFile.write('{0:0{1}X}'.format(bias, weightsPadding))
                weightsFile.write(nl)
                addrCountWeights += 1               
                
                
                
                ### Write the weight signs to the file considering 8 bits memory word ###                 
                neuron_signs = fl.w_sign[n].copy()
                                
                # neuron_signs is float64: minus = -1 and plus = 1
                neuron_signs[neuron_signs == 1] = 0     # Converts float plus to 2 complement plus 
                neuron_signs[neuron_signs == -1] = 1    # Converts float minus to 2 complement minus
                neuron_signs = neuron_signs.astype('uint8') # In order to print as hex
                i = 0
                ac = 0
                for s in neuron_signs:                            
                    if mif and i == 0:
                        signsFile.write(f'{addrCountSigns}:')

                    ac += s * (2**(ModelFxp.SIGNS_MEM_DATA_WIDTH - 1 - i))    # Converts ModelFxp.SIGNS_MEM_DATA_WIDTH signs to an hex number
                    if i == ModelFxp.SIGNS_MEM_DATA_WIDTH - 1:
                        signsFile.write('{0:0{1}X}'.format(ac, signsPadding))
                        signsFile.write(nl)
                        addrCountSigns += 1
                        ac = 0
                        i = 0
                    else:
                        i += 1

                if i > 0 and i < ModelFxp.SIGNS_MEM_DATA_WIDTH:
                    signsFile.write('{0:0{1}X}'.format(ac, signsPadding))
                    signsFile.write(nl)
                    addrCountSigns += 1

                # Bias sign. Used only to keep the same format as the weights memory
                if mif:
                    signsFile.write(f'{addrCountSigns}:') 
                
                signsFile.write('{0:0{1}X}'.format(0, signsPadding))
                signsFile.write(nl)
                addrCountSigns += 1  
                
                
                
        # Filler lines
        for _ in range(FULL_LAYER_MEM_DEPTH - addrCountWeights):
            if mif:
                weightsFile.write(f'{addrCountWeights}:')
                signsFile.write(f'{addrCountWeights}:')
                addrCountWeights += 1
            weightsFile.write('{0:0{1}X}'.format(0, weightsPadding))
            weightsFile.write(nl)
            signsFile.write('{0:0{1}X}'.format(0, signsPadding))
            signsFile.write(nl)
                       

        if mif:
            weightsFile.write(f'END;\n')
            signsFile.write(f'END;\n')

        weightsFile.close()
        signsFile.close()
        
        neurons_addr_width = np.ceil(np.log2(addrCountWeights)).astype('int32')
        print(f'\nData width: {ModelFxp.WEIGHTS_MEM_DATA_WIDTH}')
        print(f'Required neuron memories depth: {addrCountWeights} ({addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits). BRAM depth: {np.power(2, neurons_addr_width)} ({np.power(2, neurons_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits) (NEURONS_ADDR_WIDTH = {neurons_addr_width})')
        print(f'Total required memory: {len(self.FullLayers)} * {addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} = {len(self.FullLayers) * addrCountWeights * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits. Total BRAM: {len(self.FullLayers)} * {np.power(2, neurons_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} = {len(self.FullLayers) * np.power(2, neurons_addr_width) * ModelFxp.WEIGHTS_MEM_DATA_WIDTH} bits')

        
        
        return full_layer_mem_addr_width               

    def GenerateHardwiredParametersVHDL(self, input, mif = False, SerialFC = True, directory='HardwiredParameters_VHDL'):
        
        filterWeights, filterSigns, filters_addr_width  = self.MemoryImageConv(directory, mif)
        
        if SerialFC:
            full_layer_mem_addr_width = self.MemoryImageSerialFC(directory, mif)
        else:
            neuronWeights, neuronSigns, neurons_addr_width = self.MemoryImageFC(directory, mif)
                        
        # Forward until Linear
        y = input
        
        print(f'\ninput shape: {y.shape}')
        
        layer_fmap = []
        for i, l in enumerate(self.layers):
            if isinstance(l, Linear):
                y = y.flatten()
                
            y = l(y)
            
            if isinstance(l, Conv) or isinstance(l, MaxPool2d):
                print(f'{type(l)}: out shape: {y.shape}')
                layer_fmap.append({'layer' : l, 'fmap' : y})
                
                # Since fmap reduces at each Conv/MaxPool2d layer, the deepest is
                # the one after the first Conv/MaxPool2d
                if len(layer_fmap) == 1:
                    fmap_max_lines = layer_fmap[0]['fmap'].shape[2]
                    fmap_max_columns = layer_fmap[0]['fmap'].shape[3]
                elif len(layer_fmap) == 2 and isinstance(l, MaxPool2d):
                    fmap_max_lines = layer_fmap[1]['fmap'].shape[2]
                    fmap_max_columns = layer_fmap[1]['fmap'].shape[3]
        
        
        if mif:
            fmap_max_depth =  fmap_max_lines * fmap_max_columns                     
            fmaps_addr_width = np.ceil(np.log2(fmap_max_depth)).astype('uint32')
            fmaps_addr_width += 1 # +1 in case of pad(no good!)

            Path(f'{directory}/mif').mkdir(exist_ok=True, parents=True)
            featureMapFile = open(f'{directory}/mif/FeatureMap.mif', "w+")
            # mif header
            featureMapFile.write(f'DEPTH = {np.power(2,fmaps_addr_width)};\n') 
            featureMapFile.write(f'WIDTH = 21;\n')
            featureMapFile.write(f'ADDRESS_RADIX = DEC;\n')
            featureMapFile.write(f'DATA_RADIX = DEC;\n')
            featureMapFile.write(f'CONTENT\n')
            featureMapFile.write(f'BEGIN\n\n')
            featureMapFile.write(f'[0..{np.power(2,fmaps_addr_width) - 1}]: 0;\n') 
            featureMapFile.write(f'END;\n')
            
        
        Path(f'{directory}').mkdir(exist_ok=True, parents=True)
        pkg = open(f'{directory}/SkyNet_pkg.vhd', "w+")
                    
        str = """library IEEE;
use ieee.numeric_std.all;
use IEEE.std_logic_1164.all;
use std.textio.all;

package SkyNet_pkg is\n\n"""
        
        pkg.write(str)
        
        pkg.write(f'    constant FREQ_BAUD_RATE     : integer := 868; -- 115200 baud rate at 100MHz\n\n')
        
        pkg.write(f'    constant CONV_LAYERS        : integer := {len(self.ConvLayers)}; -- Number of convolution layers\n')
        
        pkg.write(f'    type conv_layers_parameters_t is array (0 to CONV_LAYERS - 1) of integer;\n')
        
        pkg.write('    constant STRIDE             : conv_layers_parameters_t := (')
        for i, cl in enumerate(self.ConvLayers):
            if (i < len(self.ConvLayers) - 1):
                pkg.write(f'{cl.stride[0]}, ')
            elif (i == len(self.ConvLayers) - 1 and len(self.ConvLayers) == 1):
                pkg.write(f'{cl.stride[0]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{cl.stride[0]});\n')
                
        pkg.write('    constant KERNEL_ORDER       : conv_layers_parameters_t := (')
        for i, cl in enumerate(self.ConvLayers):
            if (i < len(self.ConvLayers) - 1):
                pkg.write(f'{cl.w.shape[2]}, ')
            elif (i == len(self.ConvLayers) - 1 and len(self.ConvLayers) == 1):
                pkg.write(f'{cl.w.shape[2]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{cl.w.shape[2]});\n')
                
        pkg.write('    constant KERNEL_LENGTH      : conv_layers_parameters_t := (')
        for i, cl in enumerate(self.ConvLayers):
            if (i < len(self.ConvLayers) - 1):
                pkg.write(f'{cl.w.shape[2] * cl.w.shape[3]}, ')
            elif (i == len(self.ConvLayers) - 1 and len(self.ConvLayers) == 1):
                pkg.write(f'{cl.w.shape[2] * cl.w.shape[3]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{cl.w.shape[2] * cl.w.shape[3]});\n')
                           
        pkg.write('    constant CONV_INPUT_WIDTH   : conv_layers_parameters_t := (')
        
        conv_input_width = [input.shape[3]]  
        conv_input_height = [input.shape[2]] 
        for l in layer_fmap:
            if isinstance(l['layer'], Conv):
                conv_input_width.append(l['fmap'].shape[3])
                conv_input_height.append(l['fmap'].shape[2])
            else:
                # When MaxPool2d follows Conv, the input width for the next layer 
                # depends on the MaxPool2d feature map
                conv_input_width[-1] = (l['fmap'].shape[3])
                conv_input_height[-1] = (l['fmap'].shape[2])
        
        conv_input_width.pop() # The last Conv layer generates fmap to full connected
        conv_input_height.pop()
        for i, width in enumerate(conv_input_width):
            if i < len(conv_input_width) - 1:
                pkg.write(f'{width}, ')
            elif i == len(conv_input_width) - 1 and len(conv_input_width) == 1:
                pkg.write(f'{input.shape[3]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{width});\n')           
                
        pkg.write('    constant CONV_INPUT_HEIGHT  : conv_layers_parameters_t := (')
        for i, height in enumerate(conv_input_height):
            if i < len(conv_input_height) - 1:
                pkg.write(f'{height}, ')
            elif i == len(conv_input_height) - 1 and len(conv_input_height) == 1:
                pkg.write(f'{input.shape[2]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{height});\n')
        
        
        fmap_lines = []
        fmap_columns = []
        pool_kernel_order = []
        for i in range(len(layer_fmap)):
            if i < len(layer_fmap) - 1:
                if isinstance(layer_fmap[i]['layer'], Conv) and isinstance(layer_fmap[i + 1]['layer'], MaxPool2d):
                    fmap_lines.append(layer_fmap[i + 1]['fmap'].shape[2])
                    fmap_columns.append(layer_fmap[i + 1]['fmap'].shape[3])
                    pool_kernel_order.append(layer_fmap[i + 1]['layer'].k)
                elif isinstance(layer_fmap[i]['layer'], Conv):  # Last layer is Conv
                    fmap_lines.append(layer_fmap[i]['fmap'].shape[2])
                    fmap_columns.append(layer_fmap[i]['fmap'].shape[3])
                    pool_kernel_order.append(1)
                                                                 
            elif isinstance(layer_fmap[i]['layer'], Conv):  # Last layer is Conv
                fmap_lines.append(layer_fmap[i]['fmap'].shape[2])
                fmap_columns.append(layer_fmap[i]['fmap'].shape[3])
                pool_kernel_order.append(1)
        
        pkg.write('    constant FMAP_LINES         : conv_layers_parameters_t := (')
        for i, l in enumerate(fmap_lines):
            if i < len(fmap_lines) - 1:
                pkg.write(f'{l}, ')
            elif i == len(fmap_lines) - 1 and len(fmap_lines) == 1:
                pkg.write(f'{l}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{l});\n')
                
        pkg.write('    constant FMAP_COLUMNS       : conv_layers_parameters_t := (')
        for i, c in enumerate(fmap_columns):
            if i < len(fmap_columns) - 1:
                pkg.write(f'{c}, ')
            elif i == len(fmap_columns) - 1 and len(fmap_columns) == 1:
                pkg.write(f'{c}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{c});\n')
                
        pkg.write(f'    constant FMAP_MAX_LINES     : integer := {fmap_max_lines};\n')
        
        pkg.write(f'    constant FMAP_MAX_COLUMNS   : integer := {fmap_max_columns};\n')
        
        pkg.write('    constant POOL_KERNEL_ORDER  : conv_layers_parameters_t := (')
        for i, o in enumerate(pool_kernel_order): 
            if i < len(pool_kernel_order) - 1:
                pkg.write(f'{o}, ')
            elif i == len(pool_kernel_order) - 1 and len(pool_kernel_order) == 1:
                pkg.write(f'{o}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{o});\n')
                
        pkg.write('    constant PAD                : conv_layers_parameters_t := (')
        pad = []
        for i, l in enumerate(self.ConvLayers): 
            pad.append(l.pad[0])
            if i < len(self.ConvLayers) - 1:
                pkg.write(f'{l.pad[0]}, ')
            elif i == len(self.ConvLayers) - 1 and len(self.ConvLayers) == 1:
                pkg.write(f'{l.pad[0]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{l.pad[0]});\n')
        
        
        ### WARNING: Works only for padding = 0 or padding = 1 ###
        fmap_wr_addr_init = [0]
        acc = 0
        for i in reversed(range(len(fmap_lines) - 1)):
            acc += (fmap_lines[i] + pad[i + 1]) * pad[i + 1]
            
            if pad[i + 1] == 1:
                fmap_wr_addr_init.append(acc)
            else:
                fmap_wr_addr_init.append(0)
                
        fmap_wr_addr_init.reverse() 
        pkg.write('    constant FMAP_WR_ADDR_INIT  : conv_layers_parameters_t := (')
        for i, wr_addr in enumerate(fmap_wr_addr_init):
            if i < len(fmap_wr_addr_init) - 1:
                pkg.write(f'{wr_addr}, ')
            elif i == len(fmap_wr_addr_init) - 1 and len(conv_input_width) == 1:
                pkg.write(f'{wr_addr}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{wr_addr});  -- Parameter used only when PAD > 0 \n')        
        
        fmap_max_depth =  fmap_max_lines * fmap_max_columns + max(fmap_wr_addr_init)
                                  
        fmaps_addr_width = np.ceil(np.log2(fmap_max_depth)).astype('uint32')
        print(f'\nRequired feature map memories depth: {fmap_max_depth}. BRAM depth: {np.power(2,fmaps_addr_width)} (FMAPS_ADDR_WIDTH = {fmaps_addr_width})')
        print(f'Total required memory: {self.maxFilters} * {fmap_max_depth} * ACC_WIDTH bits. Total BRAM: {self.maxFilters} * {np.power(2,fmaps_addr_width)} * ACC_WIDTH bits')
        
        pkg.write(f'    constant FMAPS_ADDR_WIDTH   : integer := {fmaps_addr_width};  -- Feature map memory address bus width\n')
        
        
        
        
        pkg.write('    -- NEW_LINE includes convolutional and MaxPool2d layers stride\n')
        pkg.write('    -- Supports only square strides\n')
        pkg.write('    constant NEW_LINE           : conv_layers_parameters_t := (')
        for i, width in enumerate(conv_input_width):
            new_line = width * pool_kernel_order[i] * self.ConvLayers[i].stride[0]
            if i < len(conv_input_width) - 1:
                pkg.write(f'{new_line}, ')
            elif i == len(conv_input_width) - 1 and len(conv_input_width) == 1:
                pkg.write(f'{new_line}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{new_line}); -- Offset used to set the memory address of a new feature map line\n')
        
        pkg.write('    constant CHANNELS           : conv_layers_parameters_t := (')
        for i, cl in enumerate(self.ConvLayers):
            if (i < len(self.ConvLayers) - 1):
                pkg.write(f'{cl.w.shape[1]}, ')
            elif (i == len(self.ConvLayers) - 1 and len(self.ConvLayers) == 1):
                pkg.write(f'{cl.w.shape[1]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{cl.w.shape[1]});  -- Number of input channels in each convolution layer\n')
                       
        pkg.write('    constant FILTER_START_ADDR  : conv_layers_parameters_t := (')
        for i, addr in enumerate(self.filter_start_addr):
            if (i < len(self.filter_start_addr) - 1):
                pkg.write(f'{addr}, ')
            elif (i == len(self.filter_start_addr) - 1 and len(self.ConvLayers) == 1):
                pkg.write(f'{addr}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{addr});  -- The starting memory address of filters for each layer\n')
                     
        pkg.write('    constant CONV_LAYER_FILTERS : conv_layers_parameters_t := (')
        for i, cl in enumerate(self.ConvLayers):
            if (i < len(self.ConvLayers) - 1):
                pkg.write(f'{cl.w.shape[0]}, ')
            elif (i == len(self.ConvLayers) - 1 and len(self.ConvLayers) == 1):
                pkg.write(f'{cl.w.shape[0]}, others=>0); -- Number of filters in each convolution layer\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{cl.w.shape[0]});\n')
                
        pkg.write(f'    constant FILTERS_ADDR_WIDTH : integer := {filters_addr_width};  -- Address bus width of convolution layer filter/sign memories\n')
        
        pkg.write(f'    constant MAX_FILTERS        : integer := {self.maxFilters};  -- Number of filters in the convolution layer with more filters\n')
        
        
        
        pkg.write(f'\n    constant FULL_LAYERS        : integer := {len(self.FullLayers)};  -- Number of full connected layers\n')
        
        pkg.write(f'    type fc_layers_parameters_t is array (0 to FULL_LAYERS - 1) of integer;\n')
        
        
        if SerialFC == False:
            pkg.write('    constant WEIGHT_START_ADDR  : fc_layers_parameters_t := (')
            for i, addr in enumerate(self.weight_start_addr):            
                if (i < len(self.weight_start_addr) - 1):
                    pkg.write(f'{addr}, ')
                elif (i == len(self.weight_start_addr) - 1 and len(self.FullLayers) == 1):
                    pkg.write(f'{addr}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
                else:
                    pkg.write(f'{addr});\n')
                
        pkg.write('    constant NEURONS            : fc_layers_parameters_t := (')
        for i, n in enumerate(self.FullLayers):
            if (i < len(self.FullLayers) - 1):
                pkg.write(f'{n.w.shape[0]}, ')
            elif (i == len(self.FullLayers) - 1 and len(self.FullLayers) == 1):
                pkg.write(f'{n.w.shape[0]}, others=>0);\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'{n.w.shape[0]});  -- Number of neurons in each full connected layer\n')
        
        pkg.write(f'    constant MAX_NEURONS        : integer := {self.maxNeurons};  -- Number of neurons in the full connected layer with more neurons\n')
                                
        neurons = layer_fmap[-1]['fmap'].shape[2] * layer_fmap[-1]['fmap'].shape[3]
        pkg.write(f'    constant NEURON_INPUTS_PER_CHANNEL: integer := {neurons};  -- Number of input features in each neuron\n')
                  
        
        if SerialFC:
            pkg.write(f'    constant FULL_LAYER_ADDR_WIDTH  : integer := {full_layer_mem_addr_width};  -- Address bus width of full connected layer filter/sign memories\n')
        else:
            pkg.write(f'    constant NEURONS_ADDR_WIDTH : integer := {neurons_addr_width};\n')
        
        pkg.write(f'    constant WEIGHT_SHIFT       : integer := {self.ConvLayers[0].weight_shift};\n')
        pkg.write(f'    constant BIAS_SHIFT         : integer := {self.ConvLayers[0].bias_shift};\n')
        
        more_weigths = 0
        for cl in self.ConvLayers:
            conv_weights = cl.w.shape[2] * cl.w.shape[3]
            if conv_weights > more_weigths:
                more_weigths = conv_weights
                
        neurons_count = 0
        for fl in self.FullLayers:
            neurons_count += fl.w.shape[0]
            if fl.w.shape[1] > more_weigths:
                more_weigths = fl.w.shape[1]
                
        pkg.write(f'    constant MORE_WEIGHTS       : integer := {more_weigths};\n')
        
        pkg.write('    constant CONV_ACC_WIDTH     : integer := 20; -- Convolution layer accumulators width\n')
        pkg.write('    constant NEURON_ACC_WIDTH   : integer := 20; -- Full connected layer accumulators width\n')
        pkg.write(f'    constant WEIGHT_WIDTH       : integer := {ModelFxp.WEIGHT_WIDTH};\n')
        pkg.write('    constant DATAPATH_WIDTH     : integer := 32;\n')
        pkg.write(f'    constant WEIGHTS_PER_LINE   : integer := {ModelFxp.WEIGHTS_PER_LINE}; -- Number of weigths stored in a memory word (line)\n')
        pkg.write('    constant WEIGHTS_MEM_DATA_WIDTH   : integer := WEIGHT_WIDTH * WEIGHTS_PER_LINE;\n')

        neurons_addr_width = np.ceil(np.log2(neurons_count)).astype('uint32')        
        pkg.write(f'    constant NEURONS_ADDR_WIDTH : integer := {neurons_addr_width}; -- Address bus width of NEURONS memory\n\n')  
        
        
        
                
        str = """    -- types used in datapath
    type fmap_data_array_t  is array (0 to MAX_FILTERS - 1) of std_logic_vector(CONV_ACC_WIDTH - 1 downto 0);
    type conv_acc_array_t   is array (0 to MAX_FILTERS - 1) of SIGNED(CONV_ACC_WIDTH - 1 downto 0);

    -- multiplications
    type conv_mult_array_t      is array (0 to MAX_FILTERS - 1) of SIGNED(DATAPATH_WIDTH - 1 downto 0);

    -- weight memory addresses and weight selectors
    type filter_weights_line_array_t    is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHTS_MEM_DATA_WIDTH - 1 downto 0);
    type filter_signs_line_array_t      is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHTS_PER_LINE - 1 downto 0);
    type filter_weight_array_t          is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHT_WIDTH - 1 downto 0);
    type filter_sign_array_t            is array (0 to MAX_FILTERS - 1) of std_logic;\n\n"""
        
        pkg.write(str)
            
        if not SerialFC:
            pkg.write('    type neuron_weights_line_array_t    is array (0 to MAX_NEURONS - 1) of std_logic_vector(WEIGHTS_MEM_DATA_WIDTH - 1 downto 0);\n')
            pkg.write('    type neuron_signs_line_array_t      is array (0 to MAX_NEURONS - 1) of std_logic_vector(7 downto 0);\n\n')  
                
        pkg.write(f'    type FilterWeightsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 25);\n')
        pkg.write('    constant FILTER_WEIGTHS_FILES : FilterWeightsImageFileName_t := (\n')
        for i, f in enumerate(filterWeights):
            if i < len(filterWeights) - 1:
                pkg.write(f'        "{f}",\n')
            elif (i == len(filterWeights) - 1 and len(filterWeights) == 1):
                    str = "0" * 25
                    pkg.write(f'        "{f}", others=>"{str}");\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'        "{f}");\n\n')      
        
        pkg.write('    type FilterSignsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 23);\n')
        pkg.write('    constant FILTER_SIGNS_FILES : FilterSignsImageFileName_t := (\n')
        for i, f in enumerate(filterSigns):
            if i < len(filterSigns) - 1:
                pkg.write(f'        "{f}",\n')
            elif (i == len(filterSigns) - 1 and len(filterSigns) == 1):
                    str = "0" * 23
                    pkg.write(f'        "{f}", others=>"{str}");\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'        "{f}");\n\n')
        
        if not SerialFC:
            pkg.write('    type NeuronWeightsImageFileName_t is array (0 to MAX_NEURONS - 1) of string(1 to 25);\n')
            pkg.write('    constant NEURON_WEIGTHS_FILES : NeuronWeightsImageFileName_t := (\n')
            for i, f in enumerate(neuronWeights):
                if i < len(neuronWeights) - 1:
                    pkg.write(f'        "{f}",\n')
                else:
                    pkg.write(f'        "{f}");\n\n')        
                
            pkg.write('    type NeuronSignsImageFileName_t is array (0 to MAX_NEURONS - 1) of string(1 to 23);\n');
            pkg.write('    constant NEURON_SIGNS_FILES : NeuronSignsImageFileName_t := (\n')
            for i, f in enumerate(neuronSigns):
                if i < len(neuronSigns) - 1:
                    pkg.write(f'        "{f}",\n')
                else:
                    pkg.write(f'        "{f}");\n\n')
                 
        pkg.write('end SkyNet_pkg;')
        
        pkg.close()

    # Shifted 'inputs'
    # Parameters for 'mem_share1' implementation
    def GenerateParametersC(self, input, directory='SoftwareParameters_C'):

        Path(f'{directory}').mkdir(exist_ok=True)
        
        input_img = open(f'{directory}/input.h', 'w')
        input_img.write('#ifndef INPUT_H\n#define INPUT_H\n\n')
        input_img.write('#include <inttypes.h>\n\n')
        input_img.write('typedef int16_t input_dtype;\n\n')
        
        input_img.write(f"input_dtype input[{input.shape[0]}][{input.shape[1]}][{input.shape[2]}] = {{{{\n")
        for ch in range(input.shape[0]):
            for l in range(input.shape[1]):
                input_img.write("\t{")
                for c in range(input.shape[2]):
                    if c == input.shape[2] - 1:
                        input_img.write(f"{input[ch][l][c]}}}")
                    else:
                        input_img.write(f"{input[ch][l][c]},")
                        
                if l < input.shape[1] - 1:
                    input_img.write(",\n")


        input_img.write("\n}};\n\n")
        input_img.write('\n\n#endif')
        
        
        #create a file
        params = open(f'{directory}/parameters.h', 'w')
        params.write('#ifndef PARAMETERS_H\n#define PARAMETERS_H\n\n')
        
        params.write('#include <inttypes.h>\n\n')

        y = np.zeros((1,input.shape[0],input.shape[1], input.shape[2]))
        conv = 0
        max_filters = 0
        max_neurons = 0
        layer_fmap = []
        last_out_channels = 99999
        for l in self.layers:
            if isinstance(l, Linear):
                y = y.flatten()
                if l.w.shape[0] > max_neurons:
                    max_neurons = l.w.shape[0]
            else:                
                y = l(y)
                
            if isinstance(l, Conv) or isinstance(l, MaxPool2d) or isinstance(l, Linear):                  
                layer_fmap.append({'layer' : l, 'fmap' : y})
                
                
            if isinstance(l, Conv):
                if l.w.shape[0] > last_out_channels:
                    print("ERROR: INCREASING OUT CHANNELS NOT SUPPORTED!!!!")
                    return
                else:
                    last_out_channels = l.w.shape[0]
                    
                if l.w.shape[0] > max_filters:
                    max_filters = l.w.shape[0]
                
        params.write(f"#define INPUT_CHANNELS\t{input.shape[0]}\n")
        params.write(f"#define INPUT_LINES\t{input.shape[1]}\n")
        params.write(f"#define INPUT_COLS\t{input.shape[2]}\n\n")
        
        params.write(f"#define LAYERS\t{len(layer_fmap)}\n\n")
        
        params.write(f"#define MAX_FILTERS\t{max_filters}\n\n")
        
        params.write(f"#define MAX_NEURONS\t{max_neurons}\n\n")
        
        params.write(f"#define FC_OUT_SIZE\t{layer_fmap[-1]['layer'].w.shape[0]}\n\n")
        
        params.write(f"#define WEIGHT_SHIFT\t{layer_fmap[0]['layer'].weight_shift}\n\n")
        
        shared_fmap = layer_fmap[0]['fmap'].shape[1] * (layer_fmap[0]['fmap'].shape[2] + 2 * layer_fmap[0]['layer'].pad[0]) * (layer_fmap[0]['fmap'].shape[3] + 2 * layer_fmap[0]['layer'].pad[0])
        params.write(f"#define SHARED_FMAP_SIZE\t{shared_fmap}\n\n")
        
        str = """
#define RELU(x)	((x < 0) ? 0 : x)
#define APPLY_RELU	1
#define NO_RELU	0

#define CONV            1
#define MAX_POOL        2
#define FULL_CONNECTED  3\n\n"""
        params.write(str)
        
        params.write("typedef int16_t fmap_dtype;\n\n")       
        
        total_conv_weights = 0
        total_conv_bias = 0

        # Convolution layers filters and biases
        for i, cl in enumerate(self.ConvLayers):

            # Replace all weights equal to -1 by 0
            weights = np.where(cl.w == -1, 0, cl.w)
            signedWeights = weights * cl.w_sign
            
            total_conv_weights += weights.shape[0] * weights.shape[1] * weights.shape[2] * weights.shape[3]

            params.write(f"const int8_t filters_conv{i}[{cl.w.shape[0]}][{cl.w.shape[1]}][{cl.w.shape[2]}][{cl.w.shape[3]}] = {{\n")
            for f in range(cl.w.shape[0]):
                params.write("\t{\n")

                for k in range(cl.w.shape[1]):
                    params.write("\t\t{")

                    #params.write("\t\t\t")
                    for l in range(cl.w.shape[2]):
                        params.write("{")

                        for c in range(cl.w.shape[3]):
                            if c == cl.w.shape[3] - 1: 
                                params.write(f"{signedWeights[f][k][l][c].astype('int8')}")
                            else:
                                params.write(f"{signedWeights[f][k][l][c].astype('int8')}, ")

                        if l == cl.w.shape[2] - 1:
                            params.write("}")
                        else:
                            params.write("}, ")

                    if k == cl.w.shape[1] - 1:
                        params.write("}\n")
                    else:
                        params.write("},\n")           

                if f == cl.w.shape[0] - 1:
                    params.write("\t}\n")
                else:
                    params.write("\t},\n")

            params.write('};\n')

            params.write(f"const int16_t bias_conv{i}[{cl.b.shape[0]}] = {{")
            total_conv_bias += len(cl.b) * 2  # int16_t
            for j,b in enumerate(cl.b):
                if j == len(cl.b) - 1:
                    params.write(f"{b}")
                else:
                    params.write(f"{b}, ")
            params.write("};\n\n")

        total_fc_weights = 0
        total_fc_bias = 0
        # Full connected layers weights and biases
        for i, fl in enumerate(self.FullLayers):
            params.write(f"const int8_t weights_fc{i}[{fl.w.shape[0]}][{fl.w.shape[1]}] = {{\n")

            # Replace all weights equal to -1 by 0
            weights = np.where(fl.w == -1, 0, fl.w)
            signedWeights = (weights * fl.w_sign).astype('int8')
            
            total_fc_weights += weights.shape[0] * weights.shape[1]

            for n in range(fl.w.shape[0]):
                params.write("\t{")

                for j in range(fl.w.shape[1]):
                    if j == fl.w.shape[1] - 1:
                        params.write(f"{signedWeights[n][j]}")
                    else:
                        params.write(f"{signedWeights[n][j]}, ")

                if n == fl.w.shape[0] - 1:
                    params.write("}\n")
                else:
                    params.write("},\n")

            params.write("};\n")

            params.write(f"const int16_t bias_fc{i}[{fl.b.shape[0]}] = {{")
            total_fc_bias += len(fl.b) * 2 # int16_t
            for n in range(len(fl.b)):
                if n == len(fl.b) - 1:
                    params.write(f"{fl.b[n]}")
                else:
                    params.write(f"{fl.b[n]}, ")               

            params.write("};\n")

        
        
        str = """
struct Parameters {
    uint8_t type;
    
    /* Input feature map*/
    uint8_t fmap_in_lines;
    uint8_t fmap_in_cols;
    uint32_t fmap_in_channel_size;
    uint32_t fmap_in_channels;  /* Same as number of filter kernels */
    uint8_t pad;
    uint32_t padded_fmap;
    
    /* Output feature map*/
    uint8_t fmap_out_lines;
    uint8_t fmap_out_cols;
    uint32_t fmap_out_channel_size;
    uint32_t fmap_out_channels;
    uint32_t neurons;
    
    /* Filter features */
    const int8_t *filters;
    const int16_t *bias;
    uint8_t kernel_order;
    uint8_t kernel_size;
    uint8_t stride;
    uint16_t next_fmap_in_line;
    uint8_t apply_relu;    
};

struct Parameters parameters[LAYERS];\n\n"""
        
        params.write(str)
        
        params.write("struct Parameters *GetLayersParameters() {\n")
        
        linear = 0
        conv = 0
        
        
        for i in range(len(layer_fmap)):
            params.write(f"\n\t/* Layer {i} */\n")
            if isinstance(layer_fmap[i]['layer'], Conv):                
                params.write(f"\tparameters[{i}].type = CONV;\n")
            elif isinstance(layer_fmap[i]['layer'], MaxPool2d):
                params.write(f"\tparameters[{i}].type = MAX_POOL;\n")
            elif isinstance(layer_fmap[i]['layer'], Linear):
                params.write(f"\tparameters[{i}].type = FULL_CONNECTED;\n")
                
                
            if i == 0:
                fmap_in_lines = input.shape[1]
                fmap_in_cols = input.shape[2]
                fmap_in_channels = input.shape[0]                  
            else:                  
                fmap_in_lines = layer_fmap[i-1]['fmap'].shape[2]
                fmap_in_cols = layer_fmap[i-1]['fmap'].shape[3] 
                fmap_in_channels = layer_fmap[i-1]['fmap'].shape[1]           
            
            if isinstance(layer_fmap[i]['layer'], Conv) or isinstance(layer_fmap[i]['layer'], MaxPool2d):
                fmap_out_lines = layer_fmap[i]['fmap'].shape[2]
                fmap_out_cols = layer_fmap[i]['fmap'].shape[3] 

                    
            params.write(f"\tparameters[{i}].fmap_in_lines = {fmap_in_lines};\n")                
            params.write(f"\tparameters[{i}].fmap_in_cols = {fmap_in_cols};\n")
            params.write(f"\tparameters[{i}].fmap_in_channels = {fmap_in_channels};\n")
            params.write(f"\tparameters[{i}].fmap_in_channel_size = {fmap_in_lines * fmap_in_cols};\n\n")
                
            if isinstance(layer_fmap[i]['layer'], Conv) or isinstance(layer_fmap[i]['layer'], MaxPool2d):              
                params.write(f"\tparameters[{i}].fmap_out_lines = {fmap_out_lines};\n")                
                params.write(f"\tparameters[{i}].fmap_out_cols = {fmap_out_cols};\n")
                params.write(f"\tparameters[{i}].fmap_out_channels = {layer_fmap[i]['fmap'].shape[1]};\n")
                params.write(f"\tparameters[{i}].fmap_out_channel_size = {fmap_out_lines * fmap_out_cols};\n\n")
                
            if isinstance(layer_fmap[i]['layer'], Conv): 
                params.write(f"\tparameters[{i}].pad = {layer_fmap[i]['layer'].pad[0]};\n")
                params.write(f"\tparameters[{i}].filters = (const int8_t *)filters_conv{conv};\n")
                params.write(f"\tparameters[{i}].bias = (const int16_t *)bias_conv{conv};\n")
                conv += 1
            elif isinstance(layer_fmap[i]['layer'], Linear):
                params.write(f"\tparameters[{i}].filters = (const int8_t *)weights_fc{linear};\n")
                params.write(f"\tparameters[{i}].bias = (const int16_t *)bias_fc{linear};\n")
                params.write(f"\tparameters[{i}].neurons = {layer_fmap[i]['layer'].w.shape[0]};\n")
                linear += 1
                
                if i < len(layer_fmap) - 1 and isinstance(layer_fmap[i + 1]['layer'], Conv):
                    params.write(f"\tparameters[{i}].apply_relu = 1;\n")
                else:
                    params.write(f"\tparameters[{i}].apply_relu = 0;\n")              
                
            if isinstance(layer_fmap[i]['layer'], Conv):    
                params.write(f"\tparameters[{i}].kernel_order = {layer_fmap[i]['layer'].w.shape[2]};\n")
                params.write(f"\tparameters[{i}].kernel_size = {layer_fmap[i]['layer'].w.shape[2] * layer_fmap[i]['layer'].w.shape[2]};\n")
                params.write(f"\tparameters[{i}].stride = {layer_fmap[i]['layer'].stride[0]};\n")
                params.write(f"\tparameters[{i}].next_fmap_in_line = {layer_fmap[i]['layer'].stride[0] * fmap_in_cols};\n")
                
                if i < len(layer_fmap) - 1 and isinstance(layer_fmap[i + 1]['layer'], Conv):
                    params.write(f"\tparameters[{i}].apply_relu = 1;\n")
                else:
                    params.write(f"\tparameters[{i}].apply_relu = 0;\n")                 
        
        params.write("\n\treturn parameters;\n}\n\n")
        
        
        params.write('\n\n#endif')

        params.close()
        
        print("C parameters successfully generated!!!")
        print("\t- parameters.h")
        print("\t- input.h")
        print("\nEstimated data memory requirements:")
        array_parameters = len(layer_fmap) * 56 # 56 = sizeof(struct Parameters)
        array_acc = max_filters * 2
        array_neurons = max_neurons * 2
        input_img = input.shape[0] * input.shape[1] * input.shape[1] * 2
        print("\tRAM")
        print(f"\t\t- array parameters[{len(layer_fmap)}]: {array_parameters} Bytes")
        print(f"\t\t- shared_fmap: {shared_fmap * 2} Bytes")
        print(f"\t\t- array acc[{max_filters}] (LayerConvolution): {array_acc} Bytes")
        print(f"\t\t- array neurons[{max_neurons}] (FullConnected): {array_neurons} Bytes")
        print(f"\t\t- input: {input_img} Bytes (input_type: int16_t)")
        RAM = array_parameters + shared_fmap * 2 + array_acc + array_neurons +  input_img
        print(f"\t\t- Total: {RAM} bytes ({(RAM / 1024)} KiB)")
        print("\tROM")
        print(f"\t\t-total_conv_weights: {total_conv_weights} Bytes")
        print(f"\t\t-total_conv_bias: {total_conv_bias} Bytes")
        print(f"\t\t-total_fc_weights: {total_fc_weights} Bytes")
        print(f"\t\t-total_fc_bias: {total_fc_bias} Bytes")
        ROM = total_conv_weights + total_conv_bias + total_fc_weights + total_fc_bias
        print(f"\t\t- Total: {ROM} bytes ({(ROM / 1024)} KiB)")

    def GenerateReCoNNetParametersVHDL(self, input, mif = False, SerialFC = True, directory='ReCoNNetParameters_VHDL'):
        
        filterWeights, filterSigns, filters_addr_width  = self.MemoryImageConv(directory, mif)
        
        if SerialFC:
            full_layer_mem_addr_width = self.MemoryImageSerialFC(directory, mif)
        else:
            neuronWeights, neuronSigns, neurons_addr_width = self.MemoryImageFC(directory, mif)
                        
        # Forward until Linear
        y = input
        
        print(f'\ninput shape: {y.shape}')
        
        layer_fmap = []
        for i, l in enumerate(self.layers):
            if isinstance(l, Linear):
                y = y.flatten()
                
            y = l(y)
            
            if isinstance(l, Conv) or isinstance(l, MaxPool2d):
                print(f'{type(l)}: out shape: {y.shape}')
                layer_fmap.append({'layer' : l, 'fmap' : y})
                
                # Since fmap reduces at each Conv/MaxPool2d layer, the deepest is
                # the one after the first Conv/MaxPool2d
                if len(layer_fmap) == 1:
                    fmap_max_lines = layer_fmap[0]['fmap'].shape[2]
                    fmap_max_columns = layer_fmap[0]['fmap'].shape[3]
                elif len(layer_fmap) == 2 and isinstance(l, MaxPool2d):
                    fmap_max_lines = layer_fmap[1]['fmap'].shape[2]
                    fmap_max_columns = layer_fmap[1]['fmap'].shape[3]
        
        
        if mif:
            fmap_max_depth =  fmap_max_lines * fmap_max_columns                     
            fmaps_addr_width = np.ceil(np.log2(fmap_max_depth)).astype('uint32')
            fmaps_addr_width += 1 # +1 in case of pad(no good!)

            Path(f'{directory}/mif').mkdir(exist_ok=True, parents=True)
            featureMapFile = open(f'{directory}/mif/FeatureMap.mif', "w+")
            # mif header
            featureMapFile.write(f'DEPTH = {np.power(2,fmaps_addr_width)};\n') 
            featureMapFile.write(f'WIDTH = 21;\n')
            featureMapFile.write(f'ADDRESS_RADIX = DEC;\n')
            featureMapFile.write(f'DATA_RADIX = DEC;\n')
            featureMapFile.write(f'CONTENT\n')
            featureMapFile.write(f'BEGIN\n\n')
            featureMapFile.write(f'[0..{np.power(2,fmaps_addr_width) - 1}]: 0;\n') 
            featureMapFile.write(f'END;\n')
            
        
        Path(f'{directory}').mkdir(exist_ok=True, parents=True)
        pkg = open(f'{directory}/ReCoNNet_pkg.vhd', "w+")
                    
        str = """library IEEE;
use ieee.numeric_std.all;
use IEEE.std_logic_1164.all;
use std.textio.all;

package ReCoNNet_pkg is\n\n"""
        
        pkg.write(str)
        
        pkg.write(f'    constant FREQ_BAUD_RATE     : integer := 868; -- 115200 baud rate at 100MHz\n\n')
                
        fmap_lines = []
        fmap_columns = []
        pool_kernel_order = []
        for i in range(len(layer_fmap)):
            if i < len(layer_fmap) - 1:
                if isinstance(layer_fmap[i]['layer'], Conv) and isinstance(layer_fmap[i + 1]['layer'], MaxPool2d):
                    fmap_lines.append(layer_fmap[i + 1]['fmap'].shape[2])
                    fmap_columns.append(layer_fmap[i + 1]['fmap'].shape[3])
                    pool_kernel_order.append(layer_fmap[i + 1]['layer'].k)
                elif isinstance(layer_fmap[i]['layer'], Conv):  # Last layer is Conv
                    fmap_lines.append(layer_fmap[i]['fmap'].shape[2])
                    fmap_columns.append(layer_fmap[i]['fmap'].shape[3])
                    pool_kernel_order.append(1)
                                                                 
            elif isinstance(layer_fmap[i]['layer'], Conv):  # Last layer is Conv
                fmap_lines.append(layer_fmap[i]['fmap'].shape[2])
                fmap_columns.append(layer_fmap[i]['fmap'].shape[3])
                pool_kernel_order.append(1)
        
                
        pkg.write(f'    constant FMAP_MAX_LINES     : integer := {fmap_max_lines};\n')
        
        pkg.write(f'    constant FMAP_MAX_COLUMNS   : integer := {fmap_max_columns};\n')
        
                
        pad = []
        for l in self.ConvLayers: 
            pad.append(l.pad[0])
        
        ### WARNING: Works only for padding = 0 or padding = 1 ###
        fmap_wr_addr_init = [0]
        acc = 0
        for i in reversed(range(len(fmap_lines) - 1)):
            acc += (fmap_lines[i] + pad[i + 1]) * pad[i + 1]
            
            if pad[i + 1] == 1:
                fmap_wr_addr_init.append(acc)
            else:
                fmap_wr_addr_init.append(0)      
        
        fmap_max_depth =  fmap_max_lines * fmap_max_columns + max(fmap_wr_addr_init)
                                  
        fmaps_addr_width = np.ceil(np.log2(fmap_max_depth)).astype('uint32')
        print(f'\nRequired feature map memories depth: {fmap_max_depth}. BRAM depth: {np.power(2,fmaps_addr_width)} (FMAPS_ADDR_WIDTH = {fmaps_addr_width})')
        print(f'Total required memory: {self.maxFilters} * {fmap_max_depth} * ACC_WIDTH bits. Total BRAM: {self.maxFilters} * {np.power(2,fmaps_addr_width)} * ACC_WIDTH bits')
        
        pkg.write(f'    constant FMAPS_ADDR_WIDTH   : integer := {fmaps_addr_width};  -- Feature map memory address bus width\n')
        
        pkg.write('    -- NEW_LINE includes convolutional and MaxPool2d layers stride\n')
        pkg.write('    -- Supports only square strides\n')
                
        pkg.write(f'    constant FILTERS_ADDR_WIDTH : integer := {filters_addr_width};  -- Address bus width of convolution layer filter/sign memories\n')
        
        pkg.write(f'    constant MAX_FILTERS        : integer := {self.maxFilters};  -- Number of filters in the convolution layer with more filters\n')
                                                                        
        if SerialFC:
            pkg.write(f'    constant FULL_LAYER_ADDR_WIDTH  : integer := {full_layer_mem_addr_width};  -- Address bus width of full connected layer filter/sign memories\n')
        else:
            pkg.write(f'    constant NEURONS_ADDR_WIDTH : integer := {neurons_addr_width};\n')

        pkg.write(f'    constant CONFIG_ADDR_WIDTH  : integer := {np.max([filters_addr_width, full_layer_mem_addr_width])};\n')

                
        neurons_count = 0
        for fl in self.FullLayers:
            neurons_count += fl.w.shape[0]
        
        
        pkg.write('    constant CONV_ACC_WIDTH     : integer := 20; -- Convolution layer accumulators width\n')
        pkg.write('    constant NEURON_ACC_WIDTH   : integer := 20; -- Full connected layer accumulators width\n')
        pkg.write(f'    constant WEIGHT_WIDTH       : integer := {ModelFxp.WEIGHT_WIDTH};\n')
        pkg.write('    constant DATAPATH_WIDTH     : integer := 32;\n')
        pkg.write(f'    constant WEIGHTS_PER_LINE   : integer := {ModelFxp.WEIGHTS_PER_LINE}; -- Number of weigths stored in a memory word (line)\n')
        pkg.write('    constant WEIGHTS_MEM_DATA_WIDTH   : integer := WEIGHT_WIDTH * WEIGHTS_PER_LINE;\n')
            
        neurons_addr_width = np.ceil(np.log2(neurons_count)).astype('uint32')        
        pkg.write(f'    constant NEURONS_ADDR_WIDTH : integer := {neurons_addr_width}; -- Address bus width of NEURONS memory \n\n')  
        
        
                
        str = """    -- types used in datapath
    type fmap_data_array_t  is array (0 to MAX_FILTERS - 1) of std_logic_vector(CONV_ACC_WIDTH - 1 downto 0);
    type conv_acc_array_t   is array (0 to MAX_FILTERS - 1) of SIGNED(CONV_ACC_WIDTH - 1 downto 0);

    -- multiplications
    type conv_mult_array_t      is array (0 to MAX_FILTERS - 1) of SIGNED(DATAPATH_WIDTH - 1 downto 0);

    -- weight memory addresses and weight selectors
    type filter_weights_line_array_t    is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHTS_MEM_DATA_WIDTH - 1 downto 0);
    type filter_signs_line_array_t      is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHTS_PER_LINE - 1 downto 0);
    type filter_weight_array_t          is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHT_WIDTH - 1 downto 0);
    type filter_sign_array_t            is array (0 to MAX_FILTERS - 1) of std_logic;\n\n"""
        
        pkg.write(str)
            
        if not SerialFC:
            pkg.write('    type neuron_weights_line_array_t    is array (0 to MAX_NEURONS - 1) of std_logic_vector(WEIGHTS_MEM_DATA_WIDTH - 1 downto 0);\n')
            pkg.write('    type neuron_signs_line_array_t      is array (0 to MAX_NEURONS - 1) of std_logic_vector(7 downto 0);\n\n')  
                
        pkg.write(f'    type FilterWeightsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 25);\n')
        pkg.write('    constant FILTER_WEIGTHS_FILES : FilterWeightsImageFileName_t := (\n')
        for i, f in enumerate(filterWeights):
            if i < len(filterWeights) - 1:
                pkg.write(f'        "{f}",\n')
            elif (i == len(filterWeights) - 1 and len(filterWeights) == 1):
                    str = "0" * 25
                    pkg.write(f'        "{f}", others=>"{str}");\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'        "{f}");\n\n')      
        
        pkg.write('    type FilterSignsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 23);\n')
        pkg.write('    constant FILTER_SIGNS_FILES : FilterSignsImageFileName_t := (\n')
        for i, f in enumerate(filterSigns):
            if i < len(filterSigns) - 1:
                pkg.write(f'        "{f}",\n')
            elif (i == len(filterSigns) - 1 and len(filterSigns) == 1):
                    str = "0" * 23
                    pkg.write(f'        "{f}", others=>"{str}");\n')   # Solve length 1 array/aggregades problems in VHDL
            else:
                pkg.write(f'        "{f}");\n\n')
        
        if not SerialFC:
            pkg.write('    type NeuronWeightsImageFileName_t is array (0 to MAX_NEURONS - 1) of string(1 to 25);\n')
            pkg.write('    constant NEURON_WEIGTHS_FILES : NeuronWeightsImageFileName_t := (\n')
            for i, f in enumerate(neuronWeights):
                if i < len(neuronWeights) - 1:
                    pkg.write(f'        "{f}",\n')
                else:
                    pkg.write(f'        "{f}");\n\n')        
                
            pkg.write('    type NeuronSignsImageFileName_t is array (0 to MAX_NEURONS - 1) of string(1 to 23);\n');
            pkg.write('    constant NEURON_SIGNS_FILES : NeuronSignsImageFileName_t := (\n')
            for i, f in enumerate(neuronSigns):
                if i < len(neuronSigns) - 1:
                    pkg.write(f'        "{f}",\n')
                else:
                    pkg.write(f'        "{f}");\n\n')
                 
        pkg.write('end ReCoNNet_pkg;')
        
        pkg.close()

        #################################################
        #### Create the CNN parameters memory image  ####
        #################################################
        Path(f'{directory}').mkdir(exist_ok=True, parents=True)
        cnn_params = open(f'{directory}/CNN_params.txt', "w+")
        strSize = 2
        
        # CNN parameters                          
        cnn_params.write('{0:0{1}X}\n'.format(len(self.ConvLayers), strSize))
        cnn_params.write('{0:0{1}X}\n'.format(len(self.FullLayers), strSize))
        cnn_params.write('{0:0{1}X}\n'.format(self.ConvLayers[0].weight_shift, strSize))
        cnn_params.write('{0:0{1}X}\n'.format(self.ConvLayers[0].bias_shift, strSize))
        neurons = layer_fmap[-1]['fmap'].shape[2] * layer_fmap[-1]['fmap'].shape[3]
        cnn_params.write('{0:0{1}X}\n'.format(neurons, strSize))
        
        # Pointers to layer parameters
        paramsStart = 0x14
        cnn_params.write('{0:0{1}X}\n'.format(paramsStart, strSize)) # STRIDEs

        newParam = paramsStart
        for _ in range(14):
            newParam += len(self.ConvLayers) 
            cnn_params.write('{0:0{1}X}\n'.format(newParam, strSize)) 

        for cl in self.ConvLayers: # STRIDEs
            cnn_params.write('{0:0{1}X}\n'.format(cl.stride[0], strSize))

        for cl in self.ConvLayers: # KERNEL_ORDERs
            cnn_params.write('{0:0{1}X}\n'.format(cl.w.shape[2], strSize))

        for cl in self.ConvLayers: #KERNEL_LENGTH
            cnn_params.write('{0:0{1}X}\n'.format(cl.w.shape[2] * cl.w.shape[3], strSize))

        conv_input_width = [input.shape[3]]  
        conv_input_height = [input.shape[2]] 
        for l in layer_fmap:
            if isinstance(l['layer'], Conv):
                conv_input_width.append(l['fmap'].shape[3])
                conv_input_height.append(l['fmap'].shape[2])
            else:
                # When MaxPool2d follows Conv, the input width for the next layer 
                # depends on the MaxPool2d feature map
                conv_input_width[-1] = (l['fmap'].shape[3])
                conv_input_height[-1] = (l['fmap'].shape[2])

        conv_input_width.pop() # The last Conv layer generates fmap to full connected
        conv_input_height.pop()

        for width in conv_input_width: # CONV_INPUT_WIDTH
            cnn_params.write('{0:0{1}X}\n'.format(width, strSize))

        for height in conv_input_height: # CONV_INPUT_HEIGHT
            cnn_params.write('{0:0{1}X}\n'.format(height, strSize))

        fmap_lines = []
        fmap_columns = []
        pool_kernel_order = []
        for i in range(len(layer_fmap)):
            if i < len(layer_fmap) - 1:
                if isinstance(layer_fmap[i]['layer'], Conv) and isinstance(layer_fmap[i + 1]['layer'], MaxPool2d):
                    fmap_lines.append(layer_fmap[i + 1]['fmap'].shape[2])
                    fmap_columns.append(layer_fmap[i + 1]['fmap'].shape[3])
                    pool_kernel_order.append(layer_fmap[i + 1]['layer'].k)
                elif isinstance(layer_fmap[i]['layer'], Conv):  # Last layer is Conv
                    fmap_lines.append(layer_fmap[i]['fmap'].shape[2])
                    fmap_columns.append(layer_fmap[i]['fmap'].shape[3])
                    pool_kernel_order.append(1)
                                                                 
            elif isinstance(layer_fmap[i]['layer'], Conv):  # Last layer is Conv
                fmap_lines.append(layer_fmap[i]['fmap'].shape[2])
                fmap_columns.append(layer_fmap[i]['fmap'].shape[3])
                pool_kernel_order.append(1)

        for l in fmap_lines: # FMAP_LINES
            cnn_params.write('{0:0{1}X}\n'.format(l, strSize))

        for c in fmap_columns: # FMAP_COLUMNS
            cnn_params.write('{0:0{1}X}\n'.format(c, strSize))

        for o in pool_kernel_order: # POOL_KERNEL_ORDER
            cnn_params.write('{0:0{1}X}\n'.format(o, strSize))

        for cl in self.ConvLayers: # PADs
            cnn_params.write('{0:0{1}X}\n'.format(cl.pad[0], strSize))

        ### WARNING: Works only for padding = 0 or padding = 1 ###
        fmap_wr_addr_init = [0]
        acc = 0
        for i in reversed(range(len(fmap_lines) - 1)):
            acc += (fmap_lines[i] + pad[i + 1]) * pad[i + 1]
            
            if pad[i + 1] == 1:
                fmap_wr_addr_init.append(acc)
            else:
                fmap_wr_addr_init.append(0)
                
        fmap_wr_addr_init.reverse()
        for wr_addr in fmap_wr_addr_init: # PADs
            cnn_params.write('{0:0{1}X}\n'.format(wr_addr, strSize))

        for i, width in enumerate(conv_input_width): # NEW_LINE
            new_line = width * pool_kernel_order[i] * self.ConvLayers[i].stride[0]
            cnn_params.write('{0:0{1}X}\n'.format(new_line, strSize))

        for cl in self.ConvLayers: # CHANNELS
            cnn_params.write('{0:0{1}X}\n'.format(cl.w.shape[1], strSize))

        for addr in self.filter_start_addr: #  FILTER_START_ADDR
            cnn_params.write('{0:0{1}X}\n'.format(addr, strSize))

        for cl in self.ConvLayers: # CONV_LAYER_FILTERS
            cnn_params.write('{0:0{1}X}\n'.format(cl.w.shape[0], strSize))

        for n in self.FullLayers: # NEURONS
            cnn_params.write('{0:0{1}X}\n'.format(n.w.shape[0], strSize))
        
        cnn_params.close()


    def GenerateParametersC_mult(self, input, directory='SoftwareParameters_C'):

        Path(f'{directory}').mkdir(exist_ok=True)
        
        input_img = open(f'{directory}/input.h', 'w')
        input_img.write('#ifndef INPUT_H\n#define INPUT_H\n\n')
        input_img.write('#include <inttypes.h>\n\n')
        input_img.write('typedef int16_t input_dtype;\n\n')
        
        input_img.write(f"input_dtype input[{input.shape[0]}][{input.shape[1]}][{input.shape[2]}] = {{{{\n")
        for ch in range(input.shape[0]):
            for l in range(input.shape[1]):
                input_img.write("\t{")
                for c in range(input.shape[2]):
                    if c == input.shape[2] - 1:
                        input_img.write(f"{input[ch][l][c]}}}")
                    else:
                        input_img.write(f"{input[ch][l][c]},")
                        
                if l < input.shape[1] - 1:
                    input_img.write(",\n")


        input_img.write("\n}};\n\n")
        input_img.write('\n\n#endif')
        
        
        #create a file
        params = open(f'{directory}/parameters.h', 'w')
        params.write('#ifndef PARAMETERS_H\n#define PARAMETERS_H\n\n')
        
        params.write('#include <inttypes.h>\n\n')

        y = np.zeros((1,input.shape[0],input.shape[1], input.shape[2]))
        conv = 0
        max_filters = 0
        max_neurons = 0
        layer_fmap = []
        last_out_channels = 99999
        for l in self.layers:
            if isinstance(l, Linear):
                y = y.flatten()
                if l.w.shape[0] > max_neurons:
                    max_neurons = l.w.shape[0]
            else:                
                y = l(y)
                
            if isinstance(l, Conv) or isinstance(l, MaxPool2d) or isinstance(l, Linear):                  
                layer_fmap.append({'layer' : l, 'fmap' : y})
                
                
            if isinstance(l, Conv):
                if l.w.shape[0] > last_out_channels:
                    print("ERROR: INCREASING OUT CHANNELS NOT SUPPORTED!!!!")
                    return
                else:
                    last_out_channels = l.w.shape[0]
                    
                if l.w.shape[0] > max_filters:
                    max_filters = l.w.shape[0]
                
        params.write(f"#define INPUT_CHANNELS\t{input.shape[0]}\n")
        params.write(f"#define INPUT_LINES\t{input.shape[1]}\n")
        params.write(f"#define INPUT_COLS\t{input.shape[2]}\n\n")
        
        params.write(f"#define LAYERS\t{len(layer_fmap)}\n\n")
        
        params.write(f"#define MAX_FILTERS\t{max_filters}\n\n")
        
        params.write(f"#define MAX_NEURONS\t{max_neurons}\n\n")
        
        params.write(f"#define FC_OUT_SIZE\t{layer_fmap[-1]['layer'].w.shape[0]}\n\n")
        
        params.write(f"#define WEIGHT_SHIFT\t{layer_fmap[0]['layer'].weight_shift}\n\n")
        
        shared_fmap = layer_fmap[0]['fmap'].shape[1] * (layer_fmap[0]['fmap'].shape[2] + 2 * layer_fmap[0]['layer'].pad[0]) * (layer_fmap[0]['fmap'].shape[3] + 2 * layer_fmap[0]['layer'].pad[0])
        params.write(f"#define SHARED_FMAP_SIZE\t{shared_fmap}\n\n")
        
        str = """
#define RELU(x)	((x < 0) ? 0 : x)
#define APPLY_RELU	1
#define NO_RELU	0

#define CONV            1
#define MAX_POOL        2
#define FULL_CONNECTED  3\n\n"""
        params.write(str)
        
        params.write("typedef int16_t fmap_dtype;\n\n")       
        
        total_conv_weights = 0
        total_conv_bias = 0

        # Convolution layers filters and biases
        for i, cl in enumerate(self.ConvLayers):

            # Replace all weights equal to -1 by 0
            weights = np.where(cl.w == -1, 8, cl.w)
            signedWeights = (1 << weights) * cl.w_sign
            
            total_conv_weights += weights.shape[0] * weights.shape[1] * weights.shape[2] * weights.shape[3]

            params.write(f"const int8_t filters_conv{i}[{cl.w.shape[0]}][{cl.w.shape[1]}][{cl.w.shape[2]}][{cl.w.shape[3]}] = {{\n")
            for f in range(cl.w.shape[0]):
                params.write("\t{\n")

                for k in range(cl.w.shape[1]):
                    params.write("\t\t{")

                    #params.write("\t\t\t")
                    for l in range(cl.w.shape[2]):
                        params.write("{")

                        for c in range(cl.w.shape[3]):
                            if c == cl.w.shape[3] - 1: 
                                params.write(f"{signedWeights[f][k][l][c].astype('int8')}")
                            else:
                                params.write(f"{signedWeights[f][k][l][c].astype('int8')}, ")

                        if l == cl.w.shape[2] - 1:
                            params.write("}")
                        else:
                            params.write("}, ")

                    if k == cl.w.shape[1] - 1:
                        params.write("}\n")
                    else:
                        params.write("},\n")           

                if f == cl.w.shape[0] - 1:
                    params.write("\t}\n")
                else:
                    params.write("\t},\n")

            params.write('};\n')

            params.write(f"const int16_t bias_conv{i}[{cl.b.shape[0]}] = {{")
            total_conv_bias += len(cl.b) * 2  # int16_t
            for j,b in enumerate(cl.b):
                if j == len(cl.b) - 1:
                    params.write(f"{b}")
                else:
                    params.write(f"{b}, ")
            params.write("};\n\n")

        total_fc_weights = 0
        total_fc_bias = 0
        # Full connected layers weights and biases
        for i, fl in enumerate(self.FullLayers):
            params.write(f"const int8_t weights_fc{i}[{fl.w.shape[0]}][{fl.w.shape[1]}] = {{\n")

            # Replace all weights equal to -1 by 0
            weights = np.where(fl.w == -1, 8, fl.w)
            signedWeights = ((1 << weights) * fl.w_sign).astype('int8')
            
            total_fc_weights += weights.shape[0] * weights.shape[1]

            for n in range(fl.w.shape[0]):
                params.write("\t{")

                for j in range(fl.w.shape[1]):
                    if j == fl.w.shape[1] - 1:
                        params.write(f"{signedWeights[n][j]}")
                    else:
                        params.write(f"{signedWeights[n][j]}, ")

                if n == fl.w.shape[0] - 1:
                    params.write("}\n")
                else:
                    params.write("},\n")

            params.write("};\n")

            params.write(f"const int16_t bias_fc{i}[{fl.b.shape[0]}] = {{")
            total_fc_bias += len(fl.b) * 2 # int16_t
            for n in range(len(fl.b)):
                if n == len(fl.b) - 1:
                    params.write(f"{fl.b[n]}")
                else:
                    params.write(f"{fl.b[n]}, ")               

            params.write("};\n")

        
        
        str = """
struct Parameters {
    uint8_t type;
    
    /* Input feature map*/
    uint8_t fmap_in_lines;
    uint8_t fmap_in_cols;
    uint32_t fmap_in_channel_size;
    uint32_t fmap_in_channels;  /* Same as number of filter kernels */
    uint8_t pad;
    uint32_t padded_fmap;
    
    /* Output feature map*/
    uint8_t fmap_out_lines;
    uint8_t fmap_out_cols;
    uint32_t fmap_out_channel_size;
    uint32_t fmap_out_channels;
    uint32_t neurons;
    
    /* Filter features */
    const int8_t *filters;
    const int16_t *bias;
    uint8_t kernel_order;
    uint8_t kernel_size;
    uint8_t stride;
    uint16_t next_fmap_in_line;
    uint8_t apply_relu;    
};

struct Parameters parameters[LAYERS];\n\n"""
        
        params.write(str)
        
        params.write("struct Parameters *GetLayersParameters() {\n")
        
        linear = 0
        conv = 0
        
        
        for i in range(len(layer_fmap)):
            params.write(f"\n\t/* Layer {i} */\n")
            if isinstance(layer_fmap[i]['layer'], Conv):                
                params.write(f"\tparameters[{i}].type = CONV;\n")
            elif isinstance(layer_fmap[i]['layer'], MaxPool2d):
                params.write(f"\tparameters[{i}].type = MAX_POOL;\n")
            elif isinstance(layer_fmap[i]['layer'], Linear):
                params.write(f"\tparameters[{i}].type = FULL_CONNECTED;\n")
                
                
            if i == 0:
                fmap_in_lines = input.shape[1]
                fmap_in_cols = input.shape[2]
                fmap_in_channels = input.shape[0]                  
            else:                  
                fmap_in_lines = layer_fmap[i-1]['fmap'].shape[2]
                fmap_in_cols = layer_fmap[i-1]['fmap'].shape[3] 
                fmap_in_channels = layer_fmap[i-1]['fmap'].shape[1]           
            
            if isinstance(layer_fmap[i]['layer'], Conv) or isinstance(layer_fmap[i]['layer'], MaxPool2d):
                fmap_out_lines = layer_fmap[i]['fmap'].shape[2]
                fmap_out_cols = layer_fmap[i]['fmap'].shape[3] 

                    
            params.write(f"\tparameters[{i}].fmap_in_lines = {fmap_in_lines};\n")                
            params.write(f"\tparameters[{i}].fmap_in_cols = {fmap_in_cols};\n")
            params.write(f"\tparameters[{i}].fmap_in_channels = {fmap_in_channels};\n")
            params.write(f"\tparameters[{i}].fmap_in_channel_size = {fmap_in_lines * fmap_in_cols};\n\n")
                
            if isinstance(layer_fmap[i]['layer'], Conv) or isinstance(layer_fmap[i]['layer'], MaxPool2d):              
                params.write(f"\tparameters[{i}].fmap_out_lines = {fmap_out_lines};\n")                
                params.write(f"\tparameters[{i}].fmap_out_cols = {fmap_out_cols};\n")
                params.write(f"\tparameters[{i}].fmap_out_channels = {layer_fmap[i]['fmap'].shape[1]};\n")
                params.write(f"\tparameters[{i}].fmap_out_channel_size = {fmap_out_lines * fmap_out_cols};\n\n")
                
            if isinstance(layer_fmap[i]['layer'], Conv): 
                params.write(f"\tparameters[{i}].pad = {layer_fmap[i]['layer'].pad[0]};\n")
                params.write(f"\tparameters[{i}].filters = (const int8_t *)filters_conv{conv};\n")
                params.write(f"\tparameters[{i}].bias = (const int16_t *)bias_conv{conv};\n")
                conv += 1
            elif isinstance(layer_fmap[i]['layer'], Linear):
                params.write(f"\tparameters[{i}].filters = (const int8_t *)weights_fc{linear};\n")
                params.write(f"\tparameters[{i}].bias = (const int16_t *)bias_fc{linear};\n")
                params.write(f"\tparameters[{i}].neurons = {layer_fmap[i]['layer'].w.shape[0]};\n")
                linear += 1
                
                if i < len(layer_fmap) - 1 and isinstance(layer_fmap[i + 1]['layer'], Conv):
                    params.write(f"\tparameters[{i}].apply_relu = 1;\n")
                else:
                    params.write(f"\tparameters[{i}].apply_relu = 0;\n")              
                
            if isinstance(layer_fmap[i]['layer'], Conv):    
                params.write(f"\tparameters[{i}].kernel_order = {layer_fmap[i]['layer'].w.shape[2]};\n")
                params.write(f"\tparameters[{i}].kernel_size = {layer_fmap[i]['layer'].w.shape[2] * layer_fmap[i]['layer'].w.shape[2]};\n")
                params.write(f"\tparameters[{i}].stride = {layer_fmap[i]['layer'].stride[0]};\n")
                params.write(f"\tparameters[{i}].next_fmap_in_line = {layer_fmap[i]['layer'].stride[0] * fmap_in_cols};\n")
                
                if i < len(layer_fmap) - 1 and isinstance(layer_fmap[i + 1]['layer'], Conv):
                    params.write(f"\tparameters[{i}].apply_relu = 1;\n")
                else:
                    params.write(f"\tparameters[{i}].apply_relu = 0;\n")                 
        
        params.write("\n\treturn parameters;\n}\n\n")
        
        
        params.write('\n\n#endif')

        params.close()
        
        print("C parameters successfully generated!!!")
        print("\t- parameters.h")
        print("\t- input.h")
        print("\nEstimated data memory requirements:")
        array_parameters = len(layer_fmap) * 56 # 56 = sizeof(struct Parameters)
        array_acc = max_filters * 2
        array_neurons = max_neurons * 2
        input_img = input.shape[0] * input.shape[1] * input.shape[1] * 2
        print("\tRAM")
        print(f"\t\t- array parameters[{len(layer_fmap)}]: {array_parameters} Bytes")
        print(f"\t\t- shared_fmap: {shared_fmap * 2} Bytes")
        print(f"\t\t- array acc[{max_filters}] (LayerConvolution): {array_acc} Bytes")
        print(f"\t\t- array neurons[{max_neurons}] (FullConnected): {array_neurons} Bytes")
        print(f"\t\t- input: {input_img} Bytes (input_type: int16_t)")
        RAM = array_parameters + shared_fmap * 2 + array_acc + array_neurons +  input_img
        print(f"\t\t- Total: {RAM} bytes ({(RAM / 1024)} KiB)")
        print("\tROM")
        print(f"\t\t-total_conv_weights: {total_conv_weights} Bytes")
        print(f"\t\t-total_conv_bias: {total_conv_bias} Bytes")
        print(f"\t\t-total_fc_weights: {total_fc_weights} Bytes")
        print(f"\t\t-total_fc_bias: {total_fc_bias} Bytes")
        ROM = total_conv_weights + total_conv_bias + total_fc_weights + total_fc_bias
        print(f"\t\t- Total: {ROM} bytes ({(ROM / 1024)} KiB)")
