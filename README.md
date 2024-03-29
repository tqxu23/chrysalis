# CHRYSALIS: a novel automated EA/IA co-design methodology for autonomous things

## What is CHRYSALIS?

CHRYSALIS is an open-source tool that seeks a synergism of the energy autonomy domain and the inference autonomy domain in AuT design. The proposed methodology takes a holistic perspective, considering multiple aspects of different subsystems in an automated way.

Given a domain-specific DNN model along with its corresponding dataset, the high-level specifications of the AuT (including environment and technology constraints) as well as specific
objective demands, CHRYSALIS can automatically generate the ideal AuT solution that encompasses the configurations of energy harvester hardware (EH HW), inference hardware (Infer HW), and the dataflow of the workload. The generated solution is tailored specifically to the provided inputs, resulting in a customized and efficient AuT architecture design.

## Strcture

The code still needs to be reorganized and is planned to be optimized for subsequent incremental updates before the ISCA 2024 conference.

chrysalis  
├── models/ # models for search describers and simulators    
│   ├── components/ # components for AuT models   
│   │   ├── eh/ # energy harvesting subsystem components  
│   │   │   └── ...  
│   │   ├── insitu/ # inference subsystem components  
│   │   │   └── ...  
│   │   └── Component.py # comopnent class definition  
│   ├── DummyModel.py # basic available model   
│   ├── GammaModel.py # model for AI Accelerator-based AuT  
│   ├── iNASModel.py # model for existing AuT  
│   ├── Model.py # model class definition  
│   └── ...  
├── search/  # search scripts for quick start  
│   ├── engineA.py # search script for experimentA (Optimizing Existing AuTs with CHRYSALIS)  
│   ├── engineA.yaml # parameters for experimentA  
│   ├── engineB.py # search script for experimentB (AI Accelerator-based AuT design with CHRYSALIS)  
│   ├── engineB.yaml # parameters for experimentB  
│   └── throughput.py # basic throughput methods (to be updated)  
├── README.md  # current file  
├── README_quickstart.md  
├── requirements.txt  
└── utils.py  

## QuickStart

See `README_quickstart.md` to start search examples for artifact evaluation.

## Updates

### March 29th, 2024

Available artifact released for the Artifact Evaluation for ISCA 2024 paper.

## Acknowledgement

We reference the following existing works to build our tool.

[MAESTRO](https://github.com/maestro-project/maestro)

`Kwon Hyoukjun, Chatarasi Prasanth, Sarkar Vivek, Krishna Tushar, Pellauer Michael, and Parashar Angshuman, "MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings," in IEEE Micro, vol. 40, no. 3, pp. 20-29, 1 May-June 2020, doi: 10.1109/MM.2020.2985963.`

[GAMMA](https://github.com/maestro-project/gamma)

`Sheng-Chun Kao, and Tushar Krishna, "GAMMA: automating the HW mapping of DNN models on accelerators via genetic algorithm," in Proceedings of the 39th International Conference on Computer-Aided Design (ICCAD'20), doi: 10.1145/3400302.3415639.`

[iNAS](https://github.com/EMCLab-Sinica/Intermittent-aware-NAS)

`Hashan Roshantha Mendis, Chih-Kai Kang, and Pi-Cheng Hsiu, "Intermittent-Aware Neural Architecture Search," to appear in ACM Transactions on Embedded Computing Systems, (Integrated with IEEE/ACM CODES+ISSS 2021), doi: 10.1145/3476995.`