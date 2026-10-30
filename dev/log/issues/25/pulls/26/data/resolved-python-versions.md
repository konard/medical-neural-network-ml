# Complete resolved Python inventory

Queried PyPI JSON on 2026-10-30. `unlocked` means the old installer resolved
transitive dependencies at installation time. Registry metadata and parent
constraints are preserved in resolved-python-registry.json. Separate environments
retain upstream-required version differences; no dependency is silently omitted.

## scripts/unimedvl-requirements.txt

| Dependency | Before | Now | Latest registry release | Reason if held |
| --- | --- | --- | --- | --- |
| accelerate | 1.15.0 | 1.15.0 | 1.15.0 | current |
| annotated-doc | 0.0.5 | 0.0.5 | 0.0.5 | current |
| anyio | 4.15.1 | 4.15.1 | 4.15.1 | current |
| click | 8.5.0 | 8.5.0 | 8.5.0 | current |
| contourpy | 1.4.0 | 1.4.0 | 1.4.0 | current |
| cuda-bindings | 13.4.4 | 13.4.4 | 13.4.4 | current |
| cuda-pathfinder | 1.8.3 | 1.8.5 | 1.8.5 | current |
| cuda-toolkit | 13.0.3.0 | 13.2.2 | 13.4.3 | torch 2.15.0: ==13.2.2 |
| cycler | 0.12.1 | 0.12.1 | 0.12.1 | current |
| einops | 0.8.2 | 0.8.2 | 0.8.2 | current |
| filelock | 4.2.4 | 4.2.4 | 4.2.4 | current |
| fonttools | 4.67.0 | 4.68.0 | 4.68.0 | current |
| fsspec | 2026.9.0 | 2026.9.0 | 2026.9.0 | current |
| h11 | 0.16.0 | 0.16.0 | 0.16.0 | current |
| hf-xet | 1.7.0 | 1.8.0 | 1.8.0 | current |
| httpcore2 | 2.14.0 | 2.16.0 | 2.16.0 | current |
| httpx2 | 2.14.0 | 2.16.0 | 2.16.0 | current |
| huggingface-hub | 2.2.0 | 2.4.0 | 2.4.0 | current |
| idna | 3.20 | 3.20 | 3.20 | current |
| jinja2 | 3.1.6 | 3.1.6 | 3.1.6 | current |
| kiwisolver | 1.5.1 | 1.5.1 | 1.5.1 | current |
| markdown-it-py | 4.2.0 | 4.2.0 | 4.2.0 | current |
| markupsafe | 3.0.4 | 3.0.4 | 3.0.4 | current |
| matplotlib | 3.11.2 | 3.11.2 | 3.11.2 | current |
| mdurl | 0.1.2 | 0.1.2 | 0.1.2 | current |
| mpmath | 1.3.0 | 1.3.0 | 1.4.1 | sympy 1.14.0: <1.4,>=1.1.0 |
| networkx | 3.7 | 3.7 | 3.7 | current |
| numpy | 2.5.4 | 2.5.4 | 2.5.4 | current |
| nvidia-cublas | 13.1.1.3 | 13.4.1.3 | 13.8.2.12 | cuda-toolkit 13.2.2: ==13.4.1.3.* |
| nvidia-cuda-cccl | unlocked/new | 13.3.4.3.1 | 13.3.4.3.1 | current |
| nvidia-cuda-cupti | 13.0.85 | 13.2.86 | 13.4.92 | cuda-toolkit 13.2.2: ==13.2.86.* |
| nvidia-cuda-nvrtc | 13.0.88 | 13.2.86 | 13.4.92 | cuda-toolkit 13.2.2: ==13.2.86.* |
| nvidia-cuda-runtime | 13.0.96 | 13.2.86 | 13.4.92 | cuda-toolkit 13.2.2: ==13.2.86.* |
| nvidia-cudnn-cu13 | 9.24.0.43 | 9.26.0.51 | 9.27.0.42 | torch 2.15.0: ==9.26.0.51 |
| nvidia-cufft | 12.0.0.61 | 12.2.0.57 | 12.4.0.43 | cuda-toolkit 13.2.2: ==12.2.0.57.* |
| nvidia-cufile | 1.15.1.6 | 1.17.1.22 | 1.19.1.55 | cuda-toolkit 13.2.2: ==1.17.1.22.* |
| nvidia-cusolver | 12.0.4.66 | 12.2.0.11 | 12.3.5.3 | cuda-toolkit 13.2.2: ==12.2.0.11.* |
| nvidia-cusparse | 12.6.3.3 | 12.7.10.12 | 12.8.6.72 | cuda-toolkit 13.2.2: ==12.7.10.12.* |
| nvidia-cusparselt-cu13 | 0.8.1 | 0.8.1 | 0.10.0 | torch 2.15.0: ==0.8.1 |
| nvidia-nccl-cu13 | 2.30.7 | 2.30.7 | 2.32.3 | torch 2.15.0: ==2.30.7 |
| nvidia-nvjitlink | 13.4.92 | 13.4.92 | 13.4.92 | current |
| nvidia-nvshmem-cu13 | 3.4.5 | 3.7.2 | 3.8.0 | torch 2.15.0: ==3.7.2 |
| nvidia-nvtx | 13.0.85 | 13.2.86 | 13.4.92 | cuda-toolkit 13.2.2: ==13.2.86.* |
| opencv-python | 5.0.0.93 | 5.0.0.93 | 5.0.0.93 | current |
| packaging | 26.3 | 26.3 | 26.3 | current |
| pillow | 13.0.0 | 13.0.0 | 13.0.0 | current |
| psutil | 7.2.2 | 7.2.2 | 7.2.2 | current |
| pygments | 2.21.0 | 2.21.0 | 2.21.0 | current |
| pyparsing | 3.3.3 | 3.3.3 | 3.3.3 | current |
| python-dateutil | 2.9.0.post0 | 3.0.0 | 3.0.0 | current |
| pyyaml | 6.0.3 | 6.0.3 | 6.0.3 | current |
| regex | 2026.9.29 | 2026.9.29 | 2026.9.29 | current |
| rich | 15.0.0 | 15.0.0 | 15.0.0 | current |
| safetensors | 0.8.0 | 0.8.0 | 0.8.0 | current |
| sentencepiece | 0.2.2 | 0.2.2 | 0.2.2 | current |
| setuptools | 84.0.0 | 84.2.1 | 84.2.1 | current |
| shellingham | 1.5.4 | 1.5.4 | 1.5.4 | current |
| six | 1.17.0 | 1.17.0 | 1.17.0 | current |
| sympy | 1.14.0 | 1.14.0 | 1.14.0 | current |
| tokenizers | 0.23.3 | 0.23.3 | 0.23.3 | current |
| torch | 2.14.1 | 2.15.0 | 2.15.0 | current |
| torchvision | 0.29.1 | 0.29.2 | 0.29.2 | current |
| tqdm | 4.70.1 | 4.70.1 | 4.70.1 | current |
| transformers | 5.20.1 | 5.22.1 | 5.22.1 | current |
| triton | 3.8.0 | 3.9.0 | 3.9.0 | current |
| truststore | 0.10.5 | 0.10.5 | 0.10.5 | current |
| typer | 0.27.3 | 0.27.3 | 0.27.3 | current |
| typing-extensions | 4.16.0 | 4.16.0 | 4.16.0 | current |
| tzdata | unlocked/new | 2026.6 | 2026.6 | current |

## scripts/infra-requirements.txt

| Dependency | Before | Now | Latest registry release | Reason if held |
| --- | --- | --- | --- | --- |
| ansible-compat | unlocked/new | 26.10.0 | 26.10.0 | current |
| ansible-core | unlocked/new | 2.21.5 | 2.21.5 | current |
| ansible-lint | unlocked/new | 26.10.1 | 26.10.1 | current |
| attrs | unlocked/new | 26.1.0 | 26.1.0 | current |
| black | unlocked/new | 26.10.1 | 26.10.1 | current |
| bracex | unlocked/new | 3.2 | 3.2 | current |
| cffi | unlocked/new | 2.1.1 | 2.1.1 | current |
| click | unlocked/new | 8.5.0 | 8.5.0 | current |
| cryptography | unlocked/new | 50.0.2 | 50.0.2 | current |
| distro | unlocked/new | 1.9.0 | 1.9.0 | current |
| filelock | unlocked/new | 4.2.4 | 4.2.4 | current |
| jinja2 | unlocked/new | 3.1.6 | 3.1.6 | current |
| jsonschema | unlocked/new | 4.26.0 | 4.26.0 | current |
| jsonschema-specifications | unlocked/new | 2025.9.1 | 2025.9.1 | current |
| markupsafe | unlocked/new | 3.0.4 | 3.0.4 | current |
| mypy-extensions | unlocked/new | 1.1.0 | 1.1.0 | current |
| packaging | unlocked/new | 26.3 | 26.3 | current |
| pathspec | unlocked/new | 1.1.1 | 1.1.1 | current |
| platformdirs | unlocked/new | 4.14.0 | 4.14.0 | current |
| pycparser | unlocked/new | 3.12 | 3.12 | current |
| pytokens | unlocked/new | 0.4.1 | 0.4.1 | current |
| pyyaml | unlocked/new | 6.0.3 | 6.0.3 | current |
| referencing | unlocked/new | 0.37.0 | 0.37.0 | current |
| resolvelib | unlocked/new | 1.2.1 | 1.2.1 | current |
| rpds-py | unlocked/new | 2026.9.1 | 2026.9.1 | current |
| ruamel-yaml | unlocked/new | 0.19.1 | 0.19.1 | current |
| shellcheck-py | unlocked/new | 0.11.0.1 | 0.11.0.1 | current |
| subprocess-tee | unlocked/new | 0.4.2 | 0.4.2 | current |
| wcmatch | unlocked/new | 11.2 | 11.2 | current |
| yamllint | unlocked/new | 1.38.0 | 1.38.0 | current |

## scripts/azure-cli-requirements.txt

| Dependency | Before | Now | Latest registry release | Reason if held |
| --- | --- | --- | --- | --- |
| antlr4-python3-runtime | unlocked/new | 4.13.2 | 4.13.2 | current |
| applicationinsights | unlocked/new | 0.11.10 | 0.11.10 | current |
| argcomplete | unlocked/new | 3.5.3 | 3.7.2 | azure-cli-core 2.91.0: ~=3.5.2 |
| azure-ai-agents | unlocked/new | 1.1.0 | 1.1.0 | current |
| azure-ai-projects | unlocked/new | 1.0.0 | 2.9.0 | azure-cli 2.91.0: ~=1.0.0 |
| azure-appconfiguration | unlocked/new | 1.10.0b1 | 1.10.0 | azure-cli 2.91.0: ==1.10.0b1 |
| azure-batch | unlocked/new | 15.0.0b1 | 15.1.0 | azure-cli 2.91.0: ==15.0.0b1 |
| azure-cli | unlocked/new | 2.91.0 | 2.91.0 | current |
| azure-cli-core | unlocked/new | 2.91.0 | 2.91.0 | current |
| azure-cli-telemetry | unlocked/new | 1.1.0 | 1.1.0 | current |
| azure-common | unlocked/new | 1.1.28 | 1.1.28 | current |
| azure-core | unlocked/new | 1.39.0 | 1.41.0 | azure-cli-core 2.91.0: ~=1.39.0 |
| azure-cosmos | unlocked/new | 3.2.0 | 4.17.1 | azure-cli 2.91.0: >=3.0.2,~=3.0 |
| azure-data-tables | unlocked/new | 12.4.0 | 12.7.0 | azure-cli 2.91.0: ==12.4.0 |
| azure-datalake-store | unlocked/new | 1.0.1 | 1.0.1 | current |
| azure-keyvault-administration | unlocked/new | 4.8.0b2 | 4.7.2 | azure-cli 2.91.0: ==4.8.0b2 |
| azure-keyvault-certificates | unlocked/new | 4.7.0 | 4.12.0 | azure-cli 2.91.0: ==4.7.0 |
| azure-keyvault-keys | unlocked/new | 4.12.0b2 | 4.11.3 | azure-cli 2.91.0: ==4.12.0b2 |
| azure-keyvault-secrets | unlocked/new | 4.7.0 | 4.12.0 | azure-cli 2.91.0: ==4.7.0 |
| azure-keyvault-securitydomain | unlocked/new | 1.0.0b1 | 1.0.0b4 | azure-cli 2.91.0: ==1.0.0b1 |
| azure-mgmt-advisor | unlocked/new | 9.0.0 | 9.0.1 | azure-cli 2.91.0: ==9.0.0 |
| azure-mgmt-apimanagement | unlocked/new | 4.0.0 | 5.0.0 | azure-cli 2.91.0: ==4.0.0 |
| azure-mgmt-appconfiguration | unlocked/new | 6.0.0b2 | 5.0.0 | azure-cli 2.91.0: ==6.0.0b2 |
| azure-mgmt-appcontainers | unlocked/new | 2.0.0 | 5.1.0 | azure-cli 2.91.0: ==2.0.0 |
| azure-mgmt-applicationinsights | unlocked/new | 1.0.0 | 4.1.0 | azure-cli 2.91.0: ~=1.0.0 |
| azure-mgmt-authorization | unlocked/new | 5.0.0b2 | 4.0.0 | azure-cli 2.91.0: ==5.0.0b2 |
| azure-mgmt-batch | unlocked/new | 17.3.0 | 19.0.0 | azure-cli 2.91.0: ~=17.3.0 |
| azure-mgmt-batchai | unlocked/new | 7.0.0b1 | 7.0.0 | azure-cli 2.91.0: ==7.0.0b1 |
| azure-mgmt-billing | unlocked/new | 6.0.0 | 8.0.0 | azure-cli 2.91.0: ==6.0.0 |
| azure-mgmt-botservice | unlocked/new | 2.0.0 | 2.0.0 | current |
| azure-mgmt-cdn | unlocked/new | 12.0.0 | 14.1.0 | azure-cli 2.91.0: ==12.0.0 |
| azure-mgmt-cognitiveservices | unlocked/new | 15.0.0b4 | 14.1.0 | azure-cli 2.91.0: ==15.0.0b4 |
| azure-mgmt-compute | unlocked/new | 34.1.0 | 39.0.0 | azure-cli 2.91.0: ~=34.1.0 |
| azure-mgmt-containerinstance | unlocked/new | 10.2.0b1 | 10.1.0 | azure-cli 2.91.0: ==10.2.0b1 |
| azure-mgmt-containerregistry | unlocked/new | 15.1.0b3 | 15.0.0 | azure-cli 2.91.0: ==15.1.0b3 |
| azure-mgmt-containerregistrytasks | unlocked/new | 1.0.0b1 | 1.0.0b1 | current |
| azure-mgmt-containerservice | unlocked/new | 41.6.0 | 41.8.0 | azure-cli 2.91.0: ~=41.6.0 |
| azure-mgmt-core | unlocked/new | 1.6.0 | 1.6.0 | current |
| azure-mgmt-cosmosdb | unlocked/new | 10.0.0 | 10.0.0 | current |
| azure-mgmt-datalake-store | unlocked/new | 1.1.0b1 | 0.5.0 | azure-cli 2.91.0: ~=1.1.0b1 |
| azure-mgmt-datamigration | unlocked/new | 10.0.0 | 10.1.0 | azure-cli 2.91.0: ~=10.0.0 |
| azure-mgmt-domainregistration | unlocked/new | 1.0.0b1 | 1.0.0 | azure-cli 2.91.0: ==1.0.0b1 |
| azure-mgmt-eventgrid | unlocked/new | 10.2.0b2 | 10.4.0 | azure-cli 2.91.0: ==10.2.0b2 |
| azure-mgmt-eventhub | unlocked/new | 12.0.0 | 12.0.0 | current |
| azure-mgmt-extendedlocation | unlocked/new | 1.0.0b2 | 2.0.0 | azure-cli 2.91.0: ==1.0.0b2 |
| azure-mgmt-hdinsight | unlocked/new | 9.1.0b2 | 9.0.1 | azure-cli 2.91.0: ==9.1.0b2 |
| azure-mgmt-imagebuilder | unlocked/new | 1.3.0 | 2.0.0 | azure-cli 2.91.0: ~=1.3.0 |
| azure-mgmt-iotcentral | unlocked/new | 10.0.0b2 | 9.0.1 | azure-cli 2.91.0: ~=10.0.0b1 |
| azure-mgmt-iothub | unlocked/new | 5.0.0b1 | 4.0.0 | azure-cli 2.91.0: ==5.0.0b1 |
| azure-mgmt-iothubprovisioningservices | unlocked/new | 1.1.0 | 2.1.0 | azure-cli 2.91.0: ==1.1.0 |
| azure-mgmt-keyvault | unlocked/new | 14.0.1 | 14.1.0 | azure-cli 2.91.0: ==14.0.1 |
| azure-mgmt-loganalytics | unlocked/new | 13.0.0b4 | 14.0.0 | azure-cli 2.91.0: ==13.0.0b4 |
| azure-mgmt-managementgroups | unlocked/new | 1.0.0 | 2.0.0 | azure-cli 2.91.0: ~=1.0.0 |
| azure-mgmt-maps | unlocked/new | 2.0.0 | 2.1.0 | azure-cli 2.91.0: ~=2.0.0 |
| azure-mgmt-marketplaceordering | unlocked/new | 1.1.0 | 1.1.1 | azure-cli 2.91.0: ==1.1.0 |
| azure-mgmt-media | unlocked/new | 9.0.0 | 10.2.1 | azure-cli 2.91.0: ~=9.0 |
| azure-mgmt-monitor | unlocked/new | 7.0.0 | 7.0.0 | current |
| azure-mgmt-msi | unlocked/new | 7.1.0 | 7.1.0 | current |
| azure-mgmt-mysqlflexibleservers | unlocked/new | 1.1.0b3 | 1.0.0 | azure-cli 2.91.0: ==1.1.0b3 |
| azure-mgmt-netapp | unlocked/new | 10.1.0 | 18.0.0 | azure-cli 2.91.0: ~=10.1.0 |
| azure-mgmt-policyinsights | unlocked/new | 1.1.0b4 | 1.0.1 | azure-cli 2.91.0: ==1.1.0b4 |
| azure-mgmt-postgresqlflexibleservers | unlocked/new | 3.0.0b2 | 2.0.0 | azure-cli 2.91.0: ==3.0.0b2 |
| azure-mgmt-privatedns | unlocked/new | 1.0.0 | 2.0.0 | azure-cli 2.91.0: ~=1.0.0 |
| azure-mgmt-rdbms | unlocked/new | 10.2.0b17 | 10.1.1 | azure-cli 2.91.0: ==10.2.0b17 |
| azure-mgmt-recoveryservices | unlocked/new | 4.2.0 | 4.2.0 | current |
| azure-mgmt-recoveryservicesbackup | unlocked/new | 11.1.0 | 11.2.0 | azure-cli 2.91.0: ~=11.1.0 |
| azure-mgmt-recoveryservicesbackup-passivestamp | unlocked/new | 1.0.0b1 | 1.0.0b1 | current |
| azure-mgmt-redhatopenshift | unlocked/new | 3.0.0 | 4.0.0 | azure-cli 2.91.0: ~=3.0.0 |
| azure-mgmt-redis | unlocked/new | 14.5.0 | 14.5.0 | current |
| azure-mgmt-resource | unlocked/new | 24.0.0 | 26.0.0 | azure-cli 2.91.0: ==24.0.0 |
| azure-mgmt-resource-deployments | unlocked/new | 1.0.0b1 | 2.0.0 | azure-cli 2.91.0: ==1.0.0b1 |
| azure-mgmt-resource-deploymentscripts | unlocked/new | 1.0.0b1 | 1.0.0b1 | current |
| azure-mgmt-resource-deploymentstacks | unlocked/new | 1.0.0 | 1.0.0 | current |
| azure-mgmt-resource-templatespecs | unlocked/new | 1.0.0b1 | 1.0.0 | azure-cli 2.91.0: ==1.0.0b1 |
| azure-mgmt-search | unlocked/new | 9.2.0 | 9.2.0 | current |
| azure-mgmt-security | unlocked/new | 6.0.0 | 7.0.0 | azure-cli 2.91.0: ==6.0.0 |
| azure-mgmt-servicebus | unlocked/new | 10.0.0 | 10.0.0 | current |
| azure-mgmt-servicefabric | unlocked/new | 2.1.0 | 2.1.0 | current |
| azure-mgmt-servicefabricmanagedclusters | unlocked/new | 2.1.0b3 | 3.0.0 | azure-cli 2.91.0: ==2.1.0b3 |
| azure-mgmt-servicelinker | unlocked/new | 1.2.0b3 | 1.1.0 | azure-cli 2.91.0: ==1.2.0b3 |
| azure-mgmt-signalr | unlocked/new | 2.0.0b2 | 1.2.0 | azure-cli 2.91.0: ==2.0.0b2 |
| azure-mgmt-sql | unlocked/new | 4.0.0b22 | 4.0.0 | azure-cli 2.91.0: ==4.0.0b22 |
| azure-mgmt-sqlvirtualmachine | unlocked/new | 1.0.0b5 | 1.0.0 | azure-cli 2.91.0: ==1.0.0b5 |
| azure-mgmt-storage | unlocked/new | 25.0.0 | 25.3.0 | azure-cli 2.91.0: ==25.0.0 |
| azure-mgmt-synapse | unlocked/new | 2.1.0b5 | 2.0.0 | azure-cli 2.91.0: ==2.1.0b5 |
| azure-mgmt-trafficmanager | unlocked/new | 1.0.0 | 2.0.0 | azure-cli 2.91.0: ~=1.0.0 |
| azure-mgmt-web | unlocked/new | 11.0.0 | 11.0.1 | azure-cli 2.91.0: ==11.0.0 |
| azure-monitor-query | unlocked/new | 1.2.0 | 2.0.0 | azure-cli 2.91.0: ==1.2.0 |
| azure-storage-blob | unlocked/new | 12.29.0b1 | 12.31.1 | azure-cli 2.91.0: ==12.29.0b1 |
| azure-storage-common | unlocked/new | 1.4.2 | 2.1.0 | azure-cli 2.91.0: ~=1.4 |
| azure-storage-file-datalake | unlocked/new | 12.24.0b1 | 12.26.1 | azure-cli 2.91.0: ==12.24.0b1 |
| azure-storage-file-share | unlocked/new | 12.25.0b1 | 12.27.1 | azure-cli 2.91.0: ==12.25.0b1 |
| azure-storage-queue | unlocked/new | 12.16.0b1 | 12.18.1 | azure-cli 2.91.0: ==12.16.0b1 |
| azure-synapse-accesscontrol | unlocked/new | 0.5.0 | 0.7.0 | azure-cli 2.91.0: ~=0.5.0 |
| azure-synapse-artifacts | unlocked/new | 0.22.0 | 0.22.0 | current |
| azure-synapse-managedprivateendpoints | unlocked/new | 0.4.0 | 0.4.0 | current |
| azure-synapse-spark | unlocked/new | 0.7.0 | 0.7.0 | current |
| bcrypt | unlocked/new | 5.0.0 | 5.0.0 | current |
| certifi | unlocked/new | 2026.10.23 | 2026.10.23 | current |
| cffi | unlocked/new | 2.1.1 | 2.1.1 | current |
| chardet | unlocked/new | 5.2.0 | 7.7.0 | azure-cli 2.91.0: ~=5.2.0 |
| charset-normalizer | unlocked/new | 3.5.2 | 3.5.2 | current |
| colorama | unlocked/new | 0.4.6 | 0.4.6 | current |
| cryptography | unlocked/new | 50.0.2 | 50.0.2 | current |
| decorator | unlocked/new | 5.3.1 | 5.3.1 | current |
| deprecated | unlocked/new | 3.0.1 | 3.0.1 | current |
| distro | unlocked/new | 1.9.0 | 1.9.0 | current |
| fabric | unlocked/new | 3.2.3 | 3.2.3 | current |
| humanfriendly | unlocked/new | 10.0 | 10.0 | current |
| idna | unlocked/new | 3.20 | 3.20 | current |
| invoke | unlocked/new | 2.2.1 | 3.0.3 | fabric 3.2.3: <3.0,>=2.0 |
| isodate | unlocked/new | 0.7.2 | 0.7.2 | current |
| javaproperties | unlocked/new | 0.5.2 | 0.8.2 | azure-cli 2.91.0: ~=0.5.1 |
| jmespath | unlocked/new | 1.1.0 | 1.1.0 | current |
| jsondiff | unlocked/new | 2.0.0 | 2.2.1 | azure-cli 2.91.0: ~=2.0.0 |
| knack | unlocked/new | 0.14.0 | 0.14.0 | current |
| microsoft-security-utilities-secret-masker | unlocked/new | 1.0.0b4 | 1.0.0b4 | current |
| msal | unlocked/new | 1.39.0 | 1.39.0 | current |
| msal-extensions | unlocked/new | 1.3.1 | 1.3.1 | current |
| msrest | unlocked/new | 0.7.1 | 0.7.1 | current |
| oauthlib | unlocked/new | 4.0.0 | 4.0.0 | current |
| packaging | unlocked/new | 26.3 | 26.3 | current |
| paramiko | unlocked/new | 5.1.0 | 5.1.0 | current |
| pkginfo | unlocked/new | 2.0 | 2.0 | current |
| portalocker | unlocked/new | 2.10.1 | 4.4.2 | msal-extensions 1.3.1: <4,>=1.4 |
| psutil | unlocked/new | 7.2.2 | 7.2.2 | current |
| py-deviceid | unlocked/new | 0.1.1 | 0.1.1 | current |
| pycomposefile | unlocked/new | 0.0.34 | 0.0.34 | current |
| pycparser | unlocked/new | 3.12 | 3.12 | current |
| pygithub | unlocked/new | 1.59.1 | 2.10.0 | azure-cli 2.91.0: ~=1.38 |
| pygments | unlocked/new | 2.21.0 | 2.21.0 | current |
| pyjwt | unlocked/new | 2.15.1 | 2.15.1 | current |
| pynacl | unlocked/new | 1.6.2 | 1.6.2 | current |
| pyopenssl | unlocked/new | 26.4.0 | 26.4.0 | current |
| pysocks | unlocked/new | 1.7.1 | 1.7.1 | current |
| python-dateutil | unlocked/new | 3.0.0 | 3.0.0 | current |
| pyyaml | unlocked/new | 6.0.3 | 6.0.3 | current |
| requests | unlocked/new | 2.34.2 | 2.34.2 | current |
| requests-oauthlib | unlocked/new | 2.0.0 | 2.0.0 | current |
| scp | unlocked/new | 0.13.6 | 0.16.2 | azure-cli 2.91.0: ~=0.13.2 |
| semver | unlocked/new | 3.1.0 | 3.1.0 | current |
| six | unlocked/new | 1.17.0 | 1.17.0 | current |
| sshtunnel | unlocked/new | 0.1.5 | 0.5.1 | azure-cli 2.91.0: ~=0.1.4 |
| tabulate | unlocked/new | 0.10.0 | 0.10.0 | current |
| typing-extensions | unlocked/new | 4.16.0 | 4.16.0 | current |
| tzdata | unlocked/new | 2026.6 | 2026.6 | current |
| urllib3 | unlocked/new | 2.8.0 | 2.8.0 | current |
| websocket-client | unlocked/new | 1.8.0 | 1.9.2 | azure-cli 2.91.0: ~=1.8.0 |
| wrapt | unlocked/new | 2.5.2 | 2.5.2 | current |
| xmltodict | unlocked/new | 0.15.1 | 1.0.4 | azure-cli 2.91.0: ~=0.12 |

## dev/log/issues/25/pulls/26/data/linux-python-resolution.txt

| Dependency | Before | Now | Latest registry release | Reason if held |
| --- | --- | --- | --- | --- |
| accessible-pygments | unlocked/new | 0.0.5 | 0.0.5 | current |
| alabaster | unlocked/new | 1.0.0 | 1.0.0 | current |
| ast-serialize | unlocked/new | 0.13.1 | 0.13.1 | current |
| attrs | unlocked/new | 26.1.0 | 26.1.0 | current |
| babel | unlocked/new | 2.18.0 | 2.18.0 | current |
| beautifulsoup4 | unlocked/new | 4.15.0 | 4.15.0 | current |
| certifi | unlocked/new | 2026.10.23 | 2026.10.23 | current |
| cfgv | unlocked/new | 3.5.0 | 3.5.0 | current |
| charset-normalizer | unlocked/new | 3.5.2 | 3.5.2 | current |
| click | unlocked/new | 8.5.0 | 8.5.0 | current |
| click-log | unlocked/new | 0.4.0 | 0.4.0 | current |
| coverage | unlocked/new | 7.17.0 | 7.17.0 | current |
| distlib | unlocked/new | 0.4.4 | 0.4.4 | current |
| docutils | unlocked/new | 0.22.4 | 0.23 | myst-parser 5.1.0: <0.23,>=0.20 |
| filelock | unlocked/new | 4.2.4 | 4.2.4 | current |
| furo | unlocked/new | 2025.12.19 | 2025.12.19 | current |
| identify | unlocked/new | 2.6.20 | 2.6.20 | current |
| idna | unlocked/new | 3.20 | 3.20 | current |
| imagesize | unlocked/new | 2.0.1 | 2.0.1 | current |
| iniconfig | unlocked/new | 2.3.1 | 2.3.1 | current |
| jinja2 | unlocked/new | 3.1.6 | 3.1.6 | current |
| librt | unlocked/new | 0.16.0 | 0.16.0 | current |
| markdown-it-py | unlocked/new | 4.2.0 | 4.2.0 | current |
| markupsafe | unlocked/new | 3.0.4 | 3.0.4 | current |
| mdit-py-plugins | unlocked/new | 0.6.1 | 0.6.1 | current |
| mdurl | unlocked/new | 0.1.2 | 0.1.2 | current |
| mypy | unlocked/new | 2.4.1 | 2.4.1 | current |
| mypy-extensions | unlocked/new | 1.1.0 | 1.1.0 | current |
| myst-parser | unlocked/new | 5.1.0 | 5.1.0 | current |
| nodeenv | unlocked/new | 1.11.1 | 1.11.1 | current |
| packaging | unlocked/new | 26.3 | 26.3 | current |
| pathspec | unlocked/new | 1.1.1 | 1.1.1 | current |
| pillow | unlocked/new | 13.0.0 | 13.0.0 | current |
| platformdirs | unlocked/new | 4.14.0 | 4.14.0 | current |
| pluggy | unlocked/new | 1.7.0 | 1.7.0 | current |
| pre-commit | unlocked/new | 4.7.0 | 4.7.0 | current |
| pygments | unlocked/new | 2.21.0 | 2.21.0 | current |
| pytest | unlocked/new | 9.2.0 | 9.2.0 | current |
| pytest-asyncio | unlocked/new | 1.4.1 | 1.4.1 | current |
| pytest-cov | unlocked/new | 7.1.0 | 7.1.0 | current |
| python-discovery | unlocked/new | 1.6.7 | 1.6.7 | current |
| pyyaml | unlocked/new | 6.0.3 | 6.0.3 | current |
| requests | unlocked/new | 2.34.2 | 2.34.2 | current |
| roman-numerals | unlocked/new | 4.1.0 | 4.1.0 | current |
| ruff | unlocked/new | 0.17.3 | 0.17.3 | current |
| scriv | unlocked/new | 1.8.0 | 1.8.0 | current |
| snowballstemmer | unlocked/new | 3.1.1 | 3.1.1 | current |
| soupsieve | unlocked/new | 3.0.4 | 3.0.4 | current |
| sphinx | unlocked/new | 9.2.0 | 9.2.0 | current |
| sphinx-basic-ng | unlocked/new | 1.0.0b2 | 1.0.0b2 | current |
| sphinxcontrib-applehelp | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-devhelp | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-htmlhelp | unlocked/new | 2.1.0 | 2.1.0 | current |
| sphinxcontrib-jsmath | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-qthelp | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-serializinghtml | unlocked/new | 2.0.0 | 2.0.0 | current |
| typing-extensions | unlocked/new | 4.16.0 | 4.16.0 | current |
| urllib3 | unlocked/new | 2.8.0 | 2.8.0 | current |
| virtualenv | unlocked/new | 21.15.1 | 21.15.1 | current |

## dev/log/issues/25/pulls/26/data/macos-python-resolution.txt

| Dependency | Before | Now | Latest registry release | Reason if held |
| --- | --- | --- | --- | --- |
| accessible-pygments | unlocked/new | 0.0.5 | 0.0.5 | current |
| aiohappyeyeballs | unlocked/new | 2.7.1 | 2.7.1 | current |
| aiohttp | unlocked/new | 3.14.5 | 3.14.5 | current |
| aiosignal | unlocked/new | 1.4.0 | 1.4.0 | current |
| alabaster | unlocked/new | 1.0.0 | 1.0.0 | current |
| annotated-doc | unlocked/new | 0.0.5 | 0.0.5 | current |
| annotated-types | unlocked/new | 0.8.0 | 0.8.0 | current |
| anyio | unlocked/new | 4.15.1 | 4.15.1 | current |
| ast-serialize | unlocked/new | 0.13.1 | 0.13.1 | current |
| attrs | unlocked/new | 26.1.0 | 26.1.0 | current |
| babel | unlocked/new | 2.18.0 | 2.18.0 | current |
| beautifulsoup4 | unlocked/new | 4.15.0 | 4.15.0 | current |
| certifi | unlocked/new | 2026.10.23 | 2026.10.23 | current |
| cffi | unlocked/new | 2.1.1 | 2.1.1 | current |
| cfgv | unlocked/new | 3.5.0 | 3.5.0 | current |
| charset-normalizer | unlocked/new | 3.5.2 | 3.5.2 | current |
| click | unlocked/new | 8.5.0 | 8.5.0 | current |
| click-log | unlocked/new | 0.4.0 | 0.4.0 | current |
| coverage | unlocked/new | 7.17.0 | 7.17.0 | current |
| cryptography | unlocked/new | 50.0.2 | 50.0.2 | current |
| datasets | unlocked/new | 5.1.0 | 5.1.0 | current |
| dill | unlocked/new | 0.4.1 | 0.4.1 | current |
| distlib | unlocked/new | 0.4.4 | 0.4.4 | current |
| docutils | unlocked/new | 0.22.4 | 0.23 | myst-parser 5.1.0: <0.23,>=0.20 |
| fastapi | unlocked/new | 0.162.0 | 0.162.0 | current |
| filelock | unlocked/new | 4.2.4 | 4.2.4 | current |
| frozenlist | unlocked/new | 1.8.0 | 1.8.0 | current |
| fsspec | unlocked/new | 2026.7.0 | 2026.9.0 | datasets 5.1.0: <=2026.7.0,>=2023.1.0 |
| furo | unlocked/new | 2025.12.19 | 2025.12.19 | current |
| h11 | unlocked/new | 0.16.0 | 0.16.0 | current |
| hf-xet | unlocked/new | 1.8.0 | 1.8.0 | current |
| httpcore2 | unlocked/new | 2.16.0 | 2.16.0 | current |
| httpx2 | unlocked/new | 2.16.0 | 2.16.0 | current |
| huggingface-hub | unlocked/new | 2.4.0 | 2.4.0 | current |
| identify | unlocked/new | 2.6.20 | 2.6.20 | current |
| idna | unlocked/new | 3.20 | 3.20 | current |
| imagesize | unlocked/new | 2.0.1 | 2.0.1 | current |
| iniconfig | unlocked/new | 2.3.1 | 2.3.1 | current |
| jinja2 | unlocked/new | 3.1.6 | 3.1.6 | current |
| librt | unlocked/new | 0.16.0 | 0.16.0 | current |
| llguidance | unlocked/new | 1.9.1 | 1.9.1 | current |
| markdown-it-py | unlocked/new | 4.2.0 | 4.2.0 | current |
| markupsafe | unlocked/new | 3.0.4 | 3.0.4 | current |
| mdit-py-plugins | unlocked/new | 0.6.1 | 0.6.1 | current |
| mdurl | unlocked/new | 0.1.2 | 0.1.2 | current |
| miniaudio | unlocked/new | 1.71 | 1.71 | current |
| mlx | unlocked/new | 0.32.4 | 0.32.4 | current |
| mlx-audio | unlocked/new | 0.6.3 | 0.6.3 | current |
| mlx-metal | unlocked/new | 0.32.4 | 0.32.4 | current |
| mlx-vlm | unlocked/new | 0.7.11 | 0.7.11 | current |
| multidict | unlocked/new | 7.2.1 | 7.2.1 | current |
| multiprocess | unlocked/new | 0.70.19 | 0.70.19 | current |
| mypy | unlocked/new | 2.4.1 | 2.4.1 | current |
| mypy-extensions | unlocked/new | 1.1.0 | 1.1.0 | current |
| myst-parser | unlocked/new | 5.1.0 | 5.1.0 | current |
| nodeenv | unlocked/new | 1.11.1 | 1.11.1 | current |
| numpy | unlocked/new | 2.5.4 | 2.5.4 | current |
| opencv-python | unlocked/new | 5.0.0.93 | 5.0.0.93 | current |
| opentelemetry-api | unlocked/new | 1.45.2 | 1.45.2 | current |
| packaging | unlocked/new | 26.3 | 26.3 | current |
| pandas | unlocked/new | 3.1.0 | 3.1.0 | current |
| pathspec | unlocked/new | 1.1.1 | 1.1.1 | current |
| pillow | unlocked/new | 13.0.0 | 13.0.0 | current |
| platformdirs | unlocked/new | 4.14.0 | 4.14.0 | current |
| pluggy | unlocked/new | 1.7.0 | 1.7.0 | current |
| pre-commit | unlocked/new | 4.7.0 | 4.7.0 | current |
| propcache | unlocked/new | 0.5.4 | 0.5.4 | current |
| pyarrow | unlocked/new | 26.0.0 | 26.0.0 | current |
| pycparser | unlocked/new | 3.12 | 3.12 | current |
| pydantic | unlocked/new | 2.14.2 | 2.14.2 | current |
| pydantic-core | unlocked/new | 2.50.2 | 2.50.2 | current |
| pygments | unlocked/new | 2.21.0 | 2.21.0 | current |
| pytest | unlocked/new | 9.2.0 | 9.2.0 | current |
| pytest-asyncio | unlocked/new | 1.4.1 | 1.4.1 | current |
| pytest-cov | unlocked/new | 7.1.0 | 7.1.0 | current |
| python-dateutil | unlocked/new | 3.0.0 | 3.0.0 | current |
| python-discovery | unlocked/new | 1.6.7 | 1.6.7 | current |
| python-multipart | unlocked/new | 0.0.32 | 0.0.32 | current |
| pyyaml | unlocked/new | 6.0.3 | 6.0.3 | current |
| regex | unlocked/new | 2026.9.29 | 2026.9.29 | current |
| requests | unlocked/new | 2.34.2 | 2.34.2 | current |
| rich | unlocked/new | 15.0.0 | 15.0.0 | current |
| roman-numerals | unlocked/new | 4.1.0 | 4.1.0 | current |
| ruff | unlocked/new | 0.17.3 | 0.17.3 | current |
| safetensors | unlocked/new | 0.8.0 | 0.8.0 | current |
| scipy | unlocked/new | 1.18.1 | 1.18.1 | current |
| scriv | unlocked/new | 1.8.0 | 1.8.0 | current |
| sentencepiece | unlocked/new | 0.2.2 | 0.2.2 | current |
| shellingham | unlocked/new | 1.5.4 | 1.5.4 | current |
| six | unlocked/new | 1.17.0 | 1.17.0 | current |
| snowballstemmer | unlocked/new | 3.1.1 | 3.1.1 | current |
| sounddevice | unlocked/new | 0.5.6 | 0.5.6 | current |
| soupsieve | unlocked/new | 3.0.4 | 3.0.4 | current |
| sphinx | unlocked/new | 9.2.0 | 9.2.0 | current |
| sphinx-basic-ng | unlocked/new | 1.0.0b2 | 1.0.0b2 | current |
| sphinxcontrib-applehelp | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-devhelp | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-htmlhelp | unlocked/new | 2.1.0 | 2.1.0 | current |
| sphinxcontrib-jsmath | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-qthelp | unlocked/new | 2.0.0 | 2.0.0 | current |
| sphinxcontrib-serializinghtml | unlocked/new | 2.0.0 | 2.0.0 | current |
| starlette | unlocked/new | 1.10.0 | 1.10.0 | current |
| tokenizers | unlocked/new | 0.23.3 | 0.23.3 | current |
| tqdm | unlocked/new | 4.70.1 | 4.70.1 | current |
| transformers | unlocked/new | 5.22.1 | 5.22.1 | current |
| truststore | unlocked/new | 0.10.5 | 0.10.5 | current |
| typer | unlocked/new | 0.27.3 | 0.27.3 | current |
| typing-extensions | unlocked/new | 4.16.0 | 4.16.0 | current |
| typing-inspection | unlocked/new | 0.4.4 | 0.4.4 | current |
| tzdata | unlocked/new | 2026.6 | 2026.6 | current |
| urllib3 | unlocked/new | 2.8.0 | 2.8.0 | current |
| uvicorn | unlocked/new | 0.54.0 | 0.54.0 | current |
| virtualenv | unlocked/new | 21.15.1 | 21.15.1 | current |
| websockets | unlocked/new | 17.4 | 17.4 | current |
| xxhash | unlocked/new | 4.0.2 | 4.0.2 | current |
| yarl | unlocked/new | 1.25.1 | 1.25.1 | current |
