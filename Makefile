disco:
	docker build -f DiscoAgent.Dockerfile -t "pommerwarrior/disco" . 
crazy:
	docker build -f CrazyAgent.Dockerfile -t "pommerwarrior/crazy" . 
hybrid:
	docker build -f HybridAgent.Dockerfile -t "pommerwarrior/hybrid" . 
