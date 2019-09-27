from model import QueryEncoder, GalleryDecoder, GalleryEncoder

import torch

q = torch.rand(1, 3, 64, 64)
x = torch.rand(1, 3, 256, 256)

query_encoder   = QueryEncoder()
gallery_encoder = GalleryEncoder()
gallery_decoder = GalleryDecoder()

query_enc    = query_encoder(q)
gallery_encs = gallery_encoder(x)
print(gallery_decoder(gallery_encs, query_enc).shape)
