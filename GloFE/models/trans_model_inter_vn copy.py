import torch
import functools
import argparse

import math
from torch import nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')
from models.ctrgcn_base_p76 import Model as PoseBackbone

from models.embedding import WordAndPositionalEmbedding, PositionalEncoding
from models.inter_vn_loss import InterSampleContrastiveLoss, GloVeEmbedding
from einops import rearrange

# from models.pose_backbones import (
#     PoseBackboneWrapper,
#     SlidingWindowPoseBackbone,
#     ThinedSlidingWindowPoseBackbone,
# )
# import importlib
import models.pose_backbones as POSE_BACKBONES
from timesformer.models.vit import TimeSformer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class TransBaseModel(nn.Module):

    @staticmethod
    def add_args(parser):
        # all configuration determined before runtime should be initialized here
        # else pass as model argument
        parser.add_argument('--vocab_size', type=int, default=5725)
        parser.add_argument('--dim_embedding', type=int, default=768)
        parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'])
        parser.add_argument('--norm_first', type=str2bool, default=False, const=True, nargs='?')
        parser.add_argument('--mask_future', type=str2bool, default=True, const=True, nargs='?')
        parser.add_argument('--froze_vb', type=str2bool, default=False, const=True, nargs='?')
        # Encoder configs
        parser.add_argument('--num_enc', type=int, default=4)
        parser.add_argument('--dim_forward_enc', type=int, default=1024)
        parser.add_argument('--nhead_enc', type=int, default=8)
        parser.add_argument('--dropout_enc', type=float, default=0)
        parser.add_argument('--pe_enc', type=str2bool, default=False, const=True, nargs='?')
        parser.add_argument('--mask_enc', type=str2bool, default=False, const=True, nargs='?')

        # Decoder configs
        parser.add_argument('--num_dec', type=int, default=4)
        parser.add_argument('--dim_forward_dec', type=int, default=1024)
        parser.add_argument('--nhead_dec', type=int, default=8)
        parser.add_argument('--dropout_dec', type=float, default=0)

        # Loss configs
        parser.add_argument('--ls', type=float, default=0.0, help='Label smoothing')

        parser.add_argument('--inter_cl', type=str2bool, default=False, const=True, nargs='?')
        parser.add_argument('--inter_cl_margin', type=float, default=0.2)
        parser.add_argument('--inter_cl_alpha', type=float, default=1.0)
        ## embedding configuration for Loss
        parser.add_argument('--inter_cl_vocab', type=int, default=5563)
        parser.add_argument('--inter_cl_we_dim', type=int, default=300)
        parser.add_argument('--inter_cl_we_path', type=str, default='/mnt/workspace/slt_baseline/notebooks/uncased_filtred_glove_VN_embed.pkl')

        # Backbone config
        parser.add_argument('--pose_backbone', type=str, default='PoseBackboneWrapper')
        
        #TimeSformer Encoder config
        parser.add_argument('--frames16', type=str2bool, default=False, const=True, nargs='?')

        

        return parser

    @staticmethod
    def parse_bool(pesudo_bool):
        return pesudo_bool >= 1
    
    def train(self, mode: bool = True):
        super(TransBaseModel, self).train(mode)
        if self.args.froze_vb:
            self.visual_backbone.eval()

    def __init__(self, args, generator, sos_index):
        super(TransBaseModel, self).__init__()
        # args created by 'add_args(parser)'
        assert args.vocab_size is not None
        self.args = args
        self.generator = generator
        self.dim_embedding = args.dim_embedding
        self.vocab_size = args.vocab_size
        self.mask_future_positions = self.parse_bool(args.mask_future)
        self.norm_first = self.parse_bool(args.norm_first)
        # TODO: add SOS token
        self.sos_index = sos_index

        
        # Initialize TimeSformer as the encoder
        #Load 16 frame encoder
        if self.args.frames16 == True: 
            self.encoder = TimeSformer(
                img_size=256, 
                num_classes=self.vocab_size, 
                num_frames=16,
                attention_type='divided_space_time', 
                #patch_size=16, 
                #embed_dims=args.dim_embedding, 
                #depth=12, 
                #num_heads=8, 
                pretrained_model="/data/group1/z40575r/TimeSformer_GloFE/TimeSformer/pretrained_models/TimeSformer_divST_16x16_448_K400.pyth"
            )
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        #Load 8 frame encoder model
        elif self.args.frames16 == False:
            self.encoder = TimeSformer(
                img_size=256, 
                num_classes=self.vocab_size, 
                num_frames=8,
                attention_type='divided_space_time', 
                #patch_size=16, 
                #embed_dims=args.dim_embedding, 
                #depth=12, 
                #num_heads=8, 
                pretrained_model="/data/group1/z40575r/TimeSformer_GloFE/TimeSformer/pretrained_models/TimeSformer_divST_8x32_224_K400.pyth"
            )
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.dim_embedding, 
                nhead=args.nhead_dec,
                dim_feedforward=args.dim_forward_dec, 
                dropout=args.dropout_dec,
                batch_first=True, 
                norm_first = self.norm_first,
            ), 
            num_layers = args.num_dec,
            # Add final layer norm for pre-norm transformers.
            norm=nn.LayerNorm(self.dim_embedding) if args.norm_first else None,
        )

        if args.pe_enc:
            self.pos_embed = PositionalEncoding(max_positions = 512, dim_embed = self.dim_embedding, drop_prob = 0.0)
            # self.pos_embed = PositionalEmbeddingAndNorm(hidden_size=self.dim_embedding, max_caption_length=512)

        # embedding layer for language modeling, TODO: pass pad_index at initialization
        self.embedding = WordAndPositionalEmbedding(vocab_size=self.vocab_size, hidden_size=self.dim_embedding, max_caption_length=256, padding_idx=1)
        # self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_embedding)

        self.lm_head = nn.Linear(self.dim_embedding, self.vocab_size)
        # tie lm_head weight with embedding weights
        self.lm_head.weight = self.embedding.words.weight

        if self.args.inter_cl:
            self.glove_embed = GloVeEmbedding(
                self.args.inter_cl_vocab, 
                self.args.inter_cl_we_dim,
                self.args.inter_cl_we_path
            )
            self.inter_cl = InterSampleContrastiveLoss(self.glove_embed, margin=self.args.inter_cl_margin)
    

    
    def forward(self, x, x_length, tgt=None, tgt_length=None, vn_idxs=None, vn_len=None, phase='train'):
        assert phase in ['train', 'test'], f'Unknown phase: {phase}'
        #print("x shape", x.shape)
        # pass through TimeSformer encoder
        if phase != 'train':
            x = x[0]
        encoder_out, encoder_out_length, visual_padding_mask = self.visual_step(x, x_length, self.args.frames16, phase)
        #print("encoder out", encoder_out.shape)
        B, L, C = encoder_out.shape
        
        if phase == 'train':
            # pass through decoder 
            output_logits = self.textual_step(
                encoder_out = encoder_out,
                encoder_padding_mask = visual_padding_mask, 
                tgt = tgt, 
                tgt_length = tgt_length,
                phase = 'train'
            )

            loss, inter_cl = self.loss_step(
                logits = output_logits[:, :-1].contiguous().view(-1, self.vocab_size), 
                labels = tgt[:, 1:].contiguous().view(-1), 
                encoder_out = encoder_out,
                encoder_out_length = encoder_out_length,
                vn_idxs = vn_idxs, 
                vn_len = vn_len
            )

            output_dict = {
                'logits': output_logits,
                'loss': loss,
                'inter_cl': inter_cl,
            }

            return output_dict

        elif phase == 'test':
            # prepare `SOS` tokens for generation

            start_predictions = encoder_out.new_full((B,), self.sos_index).long()
            #print("start",start_predictions.shape)
            decoding_step = functools.partial(self.decoding_step, encoder_out, visual_padding_mask)

            predicted_caption, _ = self.generator.search(
                start_predictions, decoding_step
            )

            #This part is just used to print the validation loss
            if tgt is not None and tgt_length is not None:
                with torch.no_grad():
                    val_output_logits = self.textual_step(
                        encoder_out=encoder_out,
                        encoder_padding_mask=visual_padding_mask,
                        tgt=tgt,
                        tgt_length=tgt_length,
                        phase='test'
                    )
                    val_loss, _ = self.loss_step(
                        logits=val_output_logits[:, :-1].contiguous().view(-1, self.vocab_size),
                        labels=tgt[:, 1:].contiguous().view(-1),
                        encoder_out=encoder_out,
                        encoder_out_length=encoder_out_length,
                        vn_idxs=vn_idxs,
                        vn_len=vn_len
                    )
                    print("validation loss:" , val_loss)
            else:
                val_loss = None

            return predicted_caption


    def loss_step(
        self, logits, labels, 
        encoder_out=None, encoder_out_length=None,
        vn_idxs=None, vn_len=None
    ):

        loss = F.cross_entropy(
            logits, 
            labels, 
            ignore_index = 1, 
            label_smoothing = self.args.ls,
        )
        
        if self.args.inter_cl:
            inter_cl = self.inter_cl(encoder_out, encoder_out_length, vn_idxs, vn_len)
            loss = loss + self.args.inter_cl_alpha * inter_cl
        else:
            inter_cl = torch.tensor(0)

        return loss, inter_cl


    def process_sliding_windows(self, x, x_lengths, encoder, window_size, overlap):
        """
        x: Input tensor of shape (B, C, T, H, W)
        x_lengths: Original lengths of the sequences in the batch (B,)
        encoder: Encoder function that processes (B, C, window_size, H, W) and outputs (B, 768)
        window_size: Number of frames to process at a time
        overlap: Number of frames to overlap between windows
        """
        B, C, T, H, W = x.shape
        stride = window_size - overlap

        # Pad the input tensor to ensure all frames are covered
        padding = (window_size - stride) // 2
        x_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, padding, padding), mode='constant', value=0)

        # Calculate the number of windows
        num_windows = (T + 2 * padding - window_size) // stride + 1

        # Initialize the output tensor
        encoder_out = torch.zeros(B, num_windows, 768, device=x.device)

        # Apply the sliding window
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window = x_padded[:, :, start:end, :, :]
            encoder_out[:, i, :] = encoder(window)

        # Calculate the downsampled lengths
        encoder_out_lengths = ((x_lengths + 2 * padding - window_size) // stride) + 1

        return encoder_out, encoder_out_lengths

    
    def visual_step(self, x, x_length, toggle_16_frame=True, phase = 'train'):
        #print("x shape before permute", x.shape) # (batch, time, channel, height, width)
        x = x.permute(0,2,1,3,4) 
        #print("x shape after permute", x.shape)#(batch channel time height width)
        visual_feat, encoder_out_lengths = self.process_sliding_windows(encoder = self.encoder, x = x, x_lengths = x_length, window_size = 8, overlap = self.args.ts_overlap) #manually switch this value between 2 and 5 depending on the version

        length_scaled = torch.tensor([visual_feat.shape[1]] * visual_feat.shape[0], device=x.device)
        #print

        if self.args.pe_enc and phase == 'train':
        #if phase == 'train':
            visual_feat = self.pos_embed(visual_feat)

        if self.args.mask_enc and phase == 'train':
        #if phase == 'train':
            max_x_length, max_feat_length = x.shape[2], visual_feat.shape[1]
            ratio = (max_x_length // max_feat_length)
            # padding mask for encoder
            length_scaled = torch.ceil(x_length / ratio).type(torch.long)
            visual_padding_mask = self.make_padding_mask(visual_feat.shape[0], visual_feat.shape[1], length_scaled) # [B, max_length]
            print("visual feat time dim", visual_feat.shape[1])
            print("visual padding mask",visual_padding_mask)
        else:
            length_scaled = None
            visual_padding_mask = None

        return visual_feat, length_scaled, visual_padding_mask

    def textual_step(self, encoder_out, encoder_padding_mask, tgt, tgt_length, phase):
        max_tgt_length = tgt.shape[1]

        ones = torch.ones_like(tgt)  # [B, max_length]
        tgt_padding_mask = tgt_length.unsqueeze(1) < ones.cumsum(dim=1)  # [B, max_length]. First beam width is 1, becomes 5 after first element
        
        #if phase != 'train':
        #    print("Target shape", tgt.shape) #[Beam size, Generated sequence length]
        
        tgt_embedding = self.embedding(tgt)  # [B, L, C] [Beam size, generated sequence length, c = 768]

        if self.mask_future_positions:
        # An additive mask for masking the future (one direction).
            future_mask = self.make_future_mask(
                max_tgt_length, tgt_embedding.dtype, tgt_embedding.device
            )
        else:
            future_mask = None

        #check inputs to decoder
        
        #print("tgt embedding", tgt_embedding.shape)
        #print("encoder out", encoder_out.shape)
        #print("future mask", future_mask.shape)
        #print("tgt padding mask", tgt_padding_mask.shape)
        #if phase == 'train':
        #    print("training encoder padding mask details", encoder_padding_mask.shape)
        #    print("encoder padding mask", encoder_padding_mask)

            
        if phase != 'train':
        #    encoder_padding_mask = encoder_padding_mask[0]
            encoder_padding_mask = None
        #print("encoder padding mask details", encoder_padding_mask.shape)
        #print("encoder padding mask", encoder_padding_mask)
        decoder_out = self.decoder(
            tgt_embedding, 
            encoder_out, 
            tgt_mask = future_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = encoder_padding_mask,
        )

        output_logits = self.lm_head(decoder_out)

        return output_logits
        
    
    def decoding_step(
        self, encoder_out: torch.Tensor, encoder_padding_mask : torch.Tensor, partial_text: torch.Tensor
    ):
        B, L, C = encoder_out.shape
        # at the first step beam is just 1, increases in the following steps
        beam_size = int(partial_text.shape[0] / B)
        #print("beam size", beam_size)
        if beam_size > 1:
            #print("validation encoder out shape",encoder_out.shape) #Does not change throughout generation process
            # repeat encoder output for batched beam decoding
            encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
            encoder_out = encoder_out.view(B * beam_size, L, C)
            # repeat encoder padding mask. [B max_length]
            test_mask_enc = False
            if test_mask_enc: #self.args.mask_enc:
                encoder_padding_mask = encoder_padding_mask.unsqueeze(1).repeat(1, beam_size, 1)
                encoder_padding_mask = encoder_padding_mask.view(B * beam_size, -1)
            else:
                encoder_padding_mask = None
        if len(partial_text.size()) == 2:
            # not first timestep, pad [BOS] to partial_text
            bos_padding = partial_text.new_full((partial_text.shape[0], 1), self.sos_index).long()
            partial_text = torch.cat((bos_padding, partial_text), dim=1)

        text_lengths = torch.ones_like(partial_text)
        if len(text_lengths.size()) == 2:
            text_lengths = text_lengths.sum(1)
        else:
            # first timestep
            partial_text = partial_text.unsqueeze(1) # [B, 1]
        #print("genration monitoring", encoder_out.shape) (Beam size, video length, 768) #does not change throughout a batch 
        logits = self.textual_step(encoder_out, encoder_padding_mask, partial_text, text_lengths, phase = 'valid')
        
        # return the logits for the last timestep
        return logits[:, -1, :]

    @staticmethod
    def make_padding_mask(
        B, max_len, lengths
    ) -> torch.Tensor:

        ones = torch.ones(B, max_len).to(lengths.device) # [B, max_length]
        padding_mask = lengths.type(torch.long).unsqueeze(1) < ones.cumsum(dim=1) # [B, max_length]

        return padding_mask

    @staticmethod
    # @functools.cache # only works python >= 3.9
    def make_future_mask(
        size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Generate a mask for "future" positions. Masked positions will be negative
        infinity. This mask is critical for casual language modeling.
        """
        return torch.triu(
            torch.full((size, size), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
