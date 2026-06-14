use crate::error::TenthResult;
use crate::hir::hir::HirFnDef;

#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    F32,
    F64,
    I32,
    I64,
    Ptr(Box<KernelType>),
}

impl KernelType {
    pub fn to_c_type(&self) -> &'static str {
        match self {
            KernelType::F32 => "float",
            KernelType::F64 => "double",
            KernelType::I32 => "int32_t",
            KernelType::I64 => "int64_t",
            KernelType::Ptr(inner) => inner.to_c_type(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamDirection {
    Input,
    Output,
    InOut,
}

#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub ty: KernelType,
    pub direction: ParamDirection,
}

#[derive(Debug, Clone)]
pub struct CudaKernel {
    pub name: String,
    pub params: Vec<KernelParam>,
    pub body: String,
    pub shared_mem: usize,
}

impl CudaKernel {
    pub fn to_cuda_code(&self) -> String {
        let params_str = self
            .params
            .iter()
            .map(|p| {
                let type_str = if matches!(p.ty, KernelType::Ptr(_)) {
                    format!("{}*", p.ty.to_c_type())
                } else {
                    p.ty.to_c_type().to_string()
                };
                format!("{} {}", type_str, p.name)
            })
            .collect::<Vec<_>>()
            .join(", ");

        let shared_decl = if self.shared_mem > 0 {
            format!(
                "    extern __shared__ char s_mem[];\n    // shared memory: {} bytes\n",
                self.shared_mem
            )
        } else {
            String::new()
        };

        format!(
            "__global__ void {name}({params}) {{\n{shared}{body}}}\n",
            name = self.name,
            params = params_str,
            shared = shared_decl,
            body = self.body,
        )
    }

    pub fn from_hir_function(func: &HirFnDef) -> TenthResult<Self> {
        let name = format!("tenth_{}", func.name);
        let mut params = Vec::new();

        for (param_name, _param_ty) in &func.params {
            params.push(KernelParam {
                name: param_name.clone(),
                ty: KernelType::Ptr(Box::new(KernelType::F32)),
                direction: ParamDirection::Input,
            });
        }

        params.push(KernelParam {
            name: "out".to_string(),
            ty: KernelType::Ptr(Box::new(KernelType::F32)),
            direction: ParamDirection::Output,
        });

        params.push(KernelParam {
            name: "n".to_string(),
            ty: KernelType::I64,
            direction: ParamDirection::Input,
        });

        let body = format!(
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
             \x20   if (idx < n) {{\n\
             \x20       // kernel body for {}\n\
             \x20   }}\n",
            func.name
        );

        Ok(CudaKernel {
            name,
            params,
            body,
            shared_mem: 0,
        })
    }
}

pub fn elementwise_kernel(
    name: &str,
    input_types: &[KernelType],
    output_type: KernelType,
    n_type: KernelType,
    op: &str,
) -> CudaKernel {
    let mut params = Vec::new();

    for (i, ty) in input_types.iter().enumerate() {
        params.push(KernelParam {
            name: format!("input{}", i),
            ty: KernelType::Ptr(Box::new(ty.clone())),
            direction: ParamDirection::Input,
        });
    }

    params.push(KernelParam {
        name: "output".to_string(),
        ty: KernelType::Ptr(Box::new(output_type.clone())),
        direction: ParamDirection::Output,
    });

    params.push(KernelParam {
        name: "n".to_string(),
        ty: n_type,
        direction: ParamDirection::Input,
    });

    let input_reads: Vec<String> = (0..input_types.len())
        .map(|i| format!("input{}[idx]", i))
        .collect();

    let body = format!(
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx < n) {{\n\
         \x20       {out_ty} result = {op};\n\
         \x20       output[idx] = result;\n\
         \x20   }}\n",
        out_ty = output_type.to_c_type(),
        op = op.replace("{}", &input_reads.join(", ")),
    );

    CudaKernel {
        name: name.to_string(),
        params,
        body,
        shared_mem: 0,
    }
}

pub fn reduce_kernel(
    name: &str,
    input_type: KernelType,
    output_type: KernelType,
    n_type: KernelType,
    reduce_op: &str,
    identity: &str,
) -> CudaKernel {
    let params = vec![
        KernelParam {
            name: "input".to_string(),
            ty: KernelType::Ptr(Box::new(input_type.clone())),
            direction: ParamDirection::Input,
        },
        KernelParam {
            name: "output".to_string(),
            ty: KernelType::Ptr(Box::new(output_type)),
            direction: ParamDirection::Output,
        },
        KernelParam {
            name: "n".to_string(),
            ty: n_type,
            direction: ParamDirection::Input,
        },
    ];

    let body = format!(
        "    extern __shared__ {in_ty} sdata[];\n\
         \x20   int tid = threadIdx.x;\n\
         \x20   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   sdata[tid] = (idx < n) ? input[idx] : ({identity});\n\
         \x20   __syncthreads();\n\
         \x20   for (int s = blockDim.x / 2; s > 0; s >>= 1) {{\n\
         \x20       if (tid < s) {{\n\
         \x20           sdata[tid] = {reduce_op};\n\
         \x20       }}\n\
         \x20       __syncthreads();\n\
         \x20   }}\n\
         \x20   if (tid == 0) output[blockIdx.x] = sdata[0];\n",
        in_ty = input_type.to_c_type(),
        identity = identity,
        reduce_op = reduce_op.replace("a", "sdata[tid]").replace("b", "sdata[tid + s]"),
    );

    CudaKernel {
        name: name.to_string(),
        params,
        body,
        shared_mem: 0,
    }
}
