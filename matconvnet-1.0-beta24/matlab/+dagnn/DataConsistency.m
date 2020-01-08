classdef DataConsistency < dagnn.ElementWise
  properties
    lambda  = inf;
  end

  methods

    function outputs = forward(obj, inputs, params)
        
      % inputs	: inputs{1} 
      % input0	: inputs{2}
      % mask	: inputs{3}
      
      outputs{1} = inputs{1}.*(1 - inputs{3}) + inputs{2}.*inputs{3};
      
%       if isinf(obj.lambda)
%           outputs{1} = inputs{1}.*(1 - inputs{3}) + inputs{2}.*inputs{3};
%       else
%           outputs{1} = inputs{1}.*(1 - inputs{3}) + (inputs{1} + obj.lambda*inputs{2})/(1 + obj.lambda).*inputs{3};
%       end
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
      % inputs	: inputs{1} 
      % input0	: inputs{2}
      % mask	: inputs{3}
        
      derInputs{1}  = derOutputs{1}.*(1 - inputs{3}) ;  
%       derInputs{1}  = derOutputs{1}.*((1 - inputs{3}) + 1/(1 + obj.lambda)* inputs{3}) ;

%       if isinf(obj.lambda)
%         derInputs{1}  = derOutputs{1}.*(1 - inputs{3}) ;  
%       else
%         derInputs{1}  = derOutputs{1}.*(1 - obj.lambda/(1 + obj.lambda)*inputs{3}) ;
%       end
      derInputs{2}  = [] ;
      derInputs{3}  = [] ;
      derParams     = {} ;
        
    end
    
    function obj = DataConsistency(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
    end

  end
end
