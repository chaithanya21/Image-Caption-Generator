import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #The CNN Encoder acts as the Feature Extractor which extracts useful 
        #spatial information about the input image
        #ResNet50 is used as the pre-trained model 
        resnet = models.resnet50(pretrained=True)
        #All the parameter weights are freezed since we do not require them during the backprop
        for param in resnet.parameters():
            param.requires_grad_(False)
        #get all the layers in the resnet model except the last layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        #Add the Embedding Layer which  will be an input to the Rnn Decoder
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        
        #keep track of the hidden_size for the initialization of the hidden_state
        self.hidden_size=hidden_size 
        
        #Embedding Layer which basically turns words into a consistent size
        self.embedding_layer=nn.Embedding(vocab_size,embed_size)
        #LSTM layer with only a stack of 1 layer which takes embedded word vector as an input and 
        #ouputs an hidden state of size same as hidden_size
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers=num_layers,batch_first=True)
        self.linear=nn.Linear(hidden_size,vocab_size)
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed=self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        lstm_out,_=self.lstm(embed)
        outputs=self.linear(lstm_out)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        output_sentence = []
         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        for i in range(max_len):
            lstm_outputs,states=self.lstm(inputs,states)
            lstm_outputs=lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick).unsqueeze(1)
            
        
        
        return output_sentence
        
        
        
       
        
        
        