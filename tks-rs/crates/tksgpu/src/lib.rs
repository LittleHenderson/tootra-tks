use std::fmt;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    RequestDevice(String),
    LengthMismatch { left: usize, right: usize },
    MapRead(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no compatible GPU adapter found"),
            GpuError::RequestDevice(err) => write!(f, "failed to request device: {err}"),
            GpuError::LengthMismatch { left, right } => write!(
                f,
                "length mismatch: left has {left} elements, right has {right}"
            ),
            GpuError::MapRead(err) => write!(f, "buffer map failed: {err}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl GpuContext {
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tksgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|err| GpuError::RequestDevice(format!("{err:?}")))?;
        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn add_f32(&self, left: &[f32], right: &[f32]) -> Result<Vec<f32>, GpuError> {
        pollster::block_on(self.add_f32_async(left, right))
    }

    pub async fn add_f32_async(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        if left.len() != right.len() {
            return Err(GpuError::LengthMismatch {
                left: left.len(),
                right: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }

        let left_bytes = f32s_to_bytes(left);
        let right_bytes = f32s_to_bytes(right);
        let size = left_bytes.len() as wgpu::BufferAddress;
        let params = params_bytes(left.len() as u32);

        let left_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tksgpu-left"),
            contents: &left_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let right_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tksgpu-right"),
            contents: &right_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tksgpu-out"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tksgpu-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tksgpu-params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tksgpu-add"),
            source: wgpu::ShaderSource::Wgsl(ADD_SHADER.into()),
        });
        let bind_group_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tksgpu-add-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tksgpu-add-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tksgpu-add-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tksgpu-add-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tksgpu-add-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tksgpu-add-pass"),
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 64u32;
            let len = left.len() as u32;
            let workgroups = (len + workgroup_size - 1) / workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(GpuError::MapRead(format!("{err:?}"))),
            Err(_) => return Err(GpuError::MapRead("channel closed".to_string())),
        }

        let data = buffer_slice.get_mapped_range();
        let output = bytes_to_f32(&data);
        drop(data);
        readback.unmap();
        Ok(output)
    }
}

pub fn add_f32_cpu(left: &[f32], right: &[f32]) -> Result<Vec<f32>, GpuError> {
    if left.len() != right.len() {
        return Err(GpuError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect())
}

fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32(values: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() / 4);
    for chunk in values.chunks_exact(4) {
        out.push(f32::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
        ]));
    }
    out
}

fn params_bytes(len: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..4].copy_from_slice(&len.to_le_bytes());
    bytes
}

const ADD_SHADER: &str = r#"
struct Params {
    len: u32,
};

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < params.len) {
        out[idx] = left[idx] + right[idx];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_add_matches() {
        let left = [1.0, 2.5, 3.0];
        let right = [4.0, 1.5, -2.0];
        let out = add_f32_cpu(&left, &right).unwrap();
        assert_eq!(out, vec![5.0, 4.0, 1.0]);
    }

    #[test]
    #[ignore]
    fn gpu_add_smoke() {
        let ctx = GpuContext::new().unwrap();
        let out = ctx.add_f32(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(out, vec![4.0, 6.0]);
    }
}
