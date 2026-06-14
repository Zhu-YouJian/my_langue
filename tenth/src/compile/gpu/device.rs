use super::GpuBackend;

pub trait Device {
    fn name(&self) -> &str;
    fn device_type(&self) -> GpuBackend;
    fn memory_limit(&self) -> usize;
    fn is_available(&self) -> bool;
}

pub struct CpuDevice {
    name: String,
    memory_limit: usize,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            name: "CPU".to_string(),
            memory_limit: 16 * 1024 * 1024 * 1024, // 16 GB simulated
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for CpuDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> GpuBackend {
        GpuBackend::Cpu
    }

    fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    fn is_available(&self) -> bool {
        true
    }
}

pub struct CudaDevice {
    pub device_id: usize,
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (u32, u32),
}

impl CudaDevice {
    pub fn new(device_id: usize) -> Self {
        Self {
            device_id,
            name: format!("CUDA Device {}", device_id),
            total_memory: 24 * 1024 * 1024 * 1024, // 24 GB simulated
            compute_capability: (8, 6),
        }
    }
}

impl Device for CudaDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> GpuBackend {
        GpuBackend::Cuda
    }

    fn memory_limit(&self) -> usize {
        self.total_memory
    }

    fn is_available(&self) -> bool {
        // Simulated: always report available for code generation purposes
        true
    }
}
