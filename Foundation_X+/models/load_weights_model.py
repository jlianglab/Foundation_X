import torch
import numpy as np
import copy


def load_weights(model, args):

	if args.backbone_dir is not None:
		renaming_dict = {
			'norm.weight': 'module.backbone.classification_norm.weight',
			'norm.bias': 'module.backbone.classification_norm.bias',
		}

		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			old_value_normW = sum(model.module.backbone[0].norm.weight)
			old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			old_value_normW = sum(model.backbone[0].norm.weight)
			old_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		checkpoint = torch.load(args.backbone_dir, map_location='cpu')
		if args.init == "imagenet22k":
			state_dict = checkpoint['model']

		elif args.init == "ark":  
			state_dict = checkpoint['teacher']
			new_state_dict = {}
			for key, value in state_dict.items():
				new_key = key.replace('module.', '')  # Remove the module. prefix
				new_state_dict[new_key] = value
			state_dict = new_state_dict

		new_state_dict = {}
		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			prefix = "module.backbone.0."
		else:
			prefix = "backbone.0."
		print("[Model Info.] Model Weight Load PREFIX: ", prefix)
		for key, value in state_dict.items():
			if "head.weight" in key or "head.bias" in key:
				continue
			new_key = prefix + key
			new_state_dict[new_key] = value
		status_w = model.load_state_dict(new_state_dict, strict=False)

		new_state_dict = {}
		for old_key, new_key in renaming_dict.items():
			new_state_dict[new_key] = state_dict[old_key]

		status_w = model.load_state_dict(new_state_dict, strict=False)

		# print(status_w)
		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			new_value_normW = sum(model.module.backbone[0].norm.weight)
			new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			new_value_normW = sum(model.backbone[0].norm.weight)
			new_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		print()
		print("[Model Info.] Pretrained weights loaded for backbone:", args.backbone_dir)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_normW, new_value_normW)
		del checkpoint, state_dict, new_state_dict
		# exit(0)
		return model



	if args.resume is not None:
		print("[CHECK]  args.resume", args.resume)
		if args.resume.startswith('https'):
			checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
		else:
			checkpoint = torch.load(args.resume, map_location='cpu')

		new_state_dict = {}
		for key, value in checkpoint['model'].items():
			new_key = key.replace('module.', '')  # Remove the module. prefix
			new_state_dict[new_key] = value
		# state_dict = new_state_dict


		# state_return_msg = model.load_state_dict(checkpoint['model'], strict=False) # strict=False added for integrated model
		state_return_msg = model.load_state_dict(new_state_dict, strict=False)

		# model_summary = str(model)
		print("[C H E C K]")
		print("[Model Info.] SwinL + Dino/UperNet pretrained model loaded.")
		print(state_return_msg)
		# with open("model_summary_SwinL_DINO.txt", "w") as f:
		# 	f.write(model_summary)

		return model

def load_weights_resume(model, model_ema, optimizer, args):
		print()
		if args.resume is not None:
			print("[CHECK]  args.resume", args.resume)
		else:
			print("[CHECK]  args.pretrain_model_path", args.pretrain_model_path)
		# for k,v in model.state_dict().items():
		# 	print(k)
		# exit(0)
		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			old_value_normW = sum(model.module.backbone[0].norm.weight)
			old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			old_value_normW = sum(model.backbone[0].norm.weight)
			old_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		if args.resume is not None:
			if args.resume.startswith('https'):
				checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
			else:
				checkpoint = torch.load(args.resume, map_location='cpu')
		else:
			checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')

		# if isinstance(model, torch.nn.parallel.DistributedDataParallel):
		# 	state_return_msg = model.load_state_dict(checkpoint['model'], strict=False)
		# else:
		# 	new_state_dict = {}
		# 	for key, value in checkpoint['model'].items():
		# 		new_key = key.replace('module.', '')  # Remove the module. prefix
		# 		new_state_dict[new_key] = value
		# 	# state_dict = new_state_dict

		# 	# state_return_msg = model.load_state_dict(checkpoint['model'], strict=False) # strict=False added for integrated model
		# 	state_return_msg = model.load_state_dict(new_state_dict, strict=False)

		# if 'module' not in list(checkpoint['model'].keys())[0]:
		#     new_state_dict = {f'module.{k}': v for k, v in checkpoint['model'].items()}
		#     checkpoint['model'] = new_state_dict
		# # Load the modified state_dict into the model
		# state_return_msg = model.load_state_dict(checkpoint['model'], strict=True)

		# if args.serverC == "SOL":
		# 	if 'module' not in list(checkpoint['teacher_model'].keys())[0]:
		# 	    new_state_dict = {f'module.{k}': v for k, v in checkpoint['teacher_model'].items()}
		# 	    checkpoint['teacher_model'] = new_state_dict
		# 	# Load the modified state_dict into the model
		# 	state_return_msg = model.load_state_dict(checkpoint['teacher_model'], strict=True)
		# elif args.serverC == "DFS":
		# 	new_state_dict = {}
		# 	for key, value in checkpoint['teacher_model'].items():
		# 		# if 'segmentation_heads' in key:
		# 		# 	continue
		# 		new_key = key.replace('module.', '')  # Remove the module. prefix
		# 		new_state_dict[new_key] = value
		# 	state_return_msg = model.load_state_dict(new_state_dict, strict=True)
			
		
		state_return_msg = model.load_state_dict(checkpoint['model'], strict=True)
		print("Student Model Loaded...")
		print(state_return_msg)
		if model_ema is not None:
			state_return_msg = model_ema.load_state_dict(checkpoint['teacher_model'], strict=True)
			print("Teacher Model Loaded...")
			print(state_return_msg)
		else:
			model_ema = model
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("Optimizer Loaded...")
		
		start_epoch = checkpoint['epoch']
		print("New Starting Epoch:", start_epoch)

		### model.backbone[0].segmentation_heads[4].weight
		### model.backbone[0].segmentation_heads[4].bias
		# nn.init.uniform_(model.backbone[0].segmentation_heads[4].weight, -0.1, 0.1)
		# nn.init.uniform_(model.backbone[0].segmentation_heads[4].bias, -0.1, 0.1)
		# nn.init.uniform_(model_ema.backbone[0].segmentation_heads[4].weight, -0.1, 0.1)
		# nn.init.uniform_(model_ema.backbone[0].segmentation_heads[4].bias, -0.1, 0.1)


		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			new_value_normW = sum(model.module.backbone[0].norm.weight)
			new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			new_value_normW = sum(model.backbone[0].norm.weight)
			new_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		# model_summary = str(model)
		# print("[C H E C K]")
		print()
		print("[Model Info.] Pretrained weights loaded:", args.resume)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_normW, new_value_normW)

		# print("[Model Info.] SwinL + Dino/UperNet pretrained model loaded.")
		print()
		
		# exit(0)
		# with open("model_summary_SwinL_DINO.txt", "w") as f:
		# 	f.write(model_summary)
		return model, model_ema, optimizer, start_epoch


def load_weights_resume2(model, model_ema, optimizer_adamw, optimizer_sgd, args):
		print()
		if args.resume is not None:
			print("[CHECK]  args.resume", args.resume)
		else:
			print("[CHECK]  args.pretrain_model_path", args.pretrain_model_path)
		# for k,v in model.state_dict().items():
		# 	print(k)
		# exit(0)
		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			old_value_normW = sum(model.module.backbone[0].norm.weight)
			old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			old_value_normW = sum(model.backbone[0].norm.weight)
			old_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		if args.resume is not None:
			if args.resume.startswith('https'):
				checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
			else:
				checkpoint = torch.load(args.resume, map_location='cpu')
		else:
			checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')


		state_return_msg = model.load_state_dict(checkpoint['model'], strict=True)
		print("Student Model Loaded...")
		print(state_return_msg)
		state_return_msg = model_ema.load_state_dict(checkpoint['teacher_model'], strict=True)
		print("Teacher Model Loaded...")
		print(state_return_msg)
		optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw'])
		optimizer_sgd.load_state_dict(checkpoint['optimizer_sgd'])
		print("Optimizer Loaded...")
		
		start_epoch = checkpoint['epoch']
		print("New Starting Epoch:", start_epoch)


		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			new_value_normW = sum(model.module.backbone[0].norm.weight)
			new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			new_value_normW = sum(model.backbone[0].norm.weight)
			new_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		# model_summary = str(model)
		# print("[C H E C K]")
		print()
		print("[Model Info.] Pretrained weights loaded:", args.resume)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_normW, new_value_normW)

		# print("[Model Info.] SwinL + Dino/UperNet pretrained model loaded.")
		print()
		
		# exit(0)
		# with open("model_summary_SwinL_DINO.txt", "w") as f:
		# 	f.write(model_summary)
		return model, model_ema, optimizer_adamw, optimizer_sgd, start_epoch


# def load_weights_foundationX(model, model_ema, optimizer, args, to_load='model'): ## Load the whole FoundationX or Integrated Model
# 		print()
# 		if args.foundationX is not None:
# 			print("[CHECK]  args.foundationX", args.foundationX)

# 		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
# 			old_value_normW = sum(model.module.backbone[0].norm.weight)
# 			old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
# 		else:
# 			old_value_normW = sum(model.backbone[0].norm.weight)
# 			old_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

# 		checkpoint = torch.load(args.foundationX, map_location='cpu')
		

# 		state_return_msg = model.load_state_dict(checkpoint['model'], strict=True)
# 		print("Student Model Loaded...")
# 		print(state_return_msg)
# 		state_return_msg = model_ema.load_state_dict(checkpoint['teacher_model'], strict=True)
# 		print("Teacher Model Loaded...")
# 		print(state_return_msg)
# 		optimizer.load_state_dict(checkpoint['optimizer'])
# 		print("Optimizer Loaded...")
# 		start_epoch = checkpoint['epoch']
# 		print("New Starting Epoch:", start_epoch)


# 		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
# 			new_value_normW = sum(model.module.backbone[0].norm.weight)
# 			new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
# 		else:
# 			new_value_normW = sum(model.backbone[0].norm.weight)
# 			new_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

# 		print()
# 		print("[Model Info.] Pretrained weights loaded:", args.resume)
# 		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
# 		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_normW, new_value_normW)
# 		print()
# 		return model, model_ema


def load_weights_foundationX(model, model_ema, optimizer, args, to_load='model'): ## Load Backbone, Loc.Encoder and Seg.Decoder  || to_load = 'model' or 'teacher_model'
		print()
		if args.foundationX is not None:
			print("[CHECK]  args.foundationX", args.foundationX)

		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			old_value_normW = sum(model.module.backbone[0].norm.weight)
			old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			old_value_normW = sum(model.backbone[0].norm.weight)
			old_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		checkpoint = torch.load(args.foundationX, map_location='cpu')
		print(list(checkpoint.keys()))
		# print(checkpoint[to_load].items())
		new_state_dict = {}
		for key, value in checkpoint[to_load].items():
			### if "transformer.decoder" in key or "bbox_embed_" in key or "class_embed_" in key or "segmentation_heads" in key or "transformer.enc_out_class_embed" in key or "label_enc" in key or "class_embed" in key:
			### if "segmentation_heads" in key:
			### 	continue
			new_key = key
			new_state_dict[new_key] = value
		state_return_msg = model.load_state_dict(new_state_dict, strict=False)

		# state_return_msg = model.load_state_dict(checkpoint['model'], strict=True)
		print(to_load+" Model Loaded...")
		# print(state_return_msg)
		# state_return_msg = model_ema.load_state_dict(checkpoint['teacher_model'], strict=True)
		# print("Teacher Model Loaded...")
		# print(state_return_msg)


		if isinstance(model, torch.nn.parallel.DistributedDataParallel):
			new_value_normW = sum(model.module.backbone[0].norm.weight)
			new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)
		else:
			new_value_normW = sum(model.backbone[0].norm.weight)
			new_value_layernorm = sum(model.backbone[0].layers[2].blocks[11].norm1.weight)

		print()
		print("[Model Info.] Pretrained weights loaded:", args.foundationX)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
		print("[Model CHECK] Loaded backbone weights -- Before & After --  norm.weight and norm.bias.", old_value_normW, new_value_normW)
		print()



		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Backbone (Swin-B) with Classification Heads: {total_params}")

		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('backbone' in name) and ('segmentation_' not in name) and ('classification_heads' not in name): # BACKBONE only
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Backbone (Swin-B): {total_params}")

		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('classification_heads' in name): # Classification Heads (11)
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Classification Heads (total 11): {total_params}")

		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('transformer.encoder' in name): # Localization Enc
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Localization Encoder: {total_params}")

		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('transformer.decoder' in name): # Localization Dec
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Localization Decoders (total 6): {total_params}")

		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('segmentation_' in name) and ('segmentation_heads' not in name): # Segmentation Dec only
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Segmentation Decoders: {total_params}")

		for name, param in model.named_parameters():
			param.requires_grad = False
		for name, param in model.named_parameters():
			if ('segmentation_heads' in name): # Segmentation Heads only
				param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Segmentation Heads (total 3): {total_params}")


		for name, param in model.named_parameters():
			param.requires_grad = True
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total Trainable Parameters - Foundation X: {total_params}")

		# exit()

		return model, copy.deepcopy(model)