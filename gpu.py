"""Headless Vulkan device for vabs compute, plus reusable command recordings.

vabs' ``Context`` presents to a Win32 window. Training needs only a compute
queue, buffer device addresses and 64-bit integers (for ``BufferRef``), so
this context creates no surface or swapchain. It supplies the attributes
that ``vabs.memory.Arena``, ``vabs.pipeline.ComputePipeline`` and
``vabs.renderer.Frame`` use (same approach as ``pbd_gpu/_device.py``).

``Recording`` records a ``vabs.renderer.Frame`` into a command buffer once;
``submit()`` runs it again as often as needed. Every training step is the
same sequence of dispatches over the same buffers, so only host-visible
inputs (tokens, learning rate) change between submissions.
"""
import cvk
import vk
from vabs import _native, memory_stats
from vabs.renderer import Frame


class ComputeContext:
    """Instance, device, compute queue and command pool; no presentation."""

    def __init__(self, app_name='vabs_gpt'):
        self._validation = False
        self.instance = self.device = self.command_pool = None
        self.swapchain, self.swapchain_images, self.swapchain_extent = None, [], (1, 1)
        self._arenas, self._on_destroy = [], []
        self.memory_stats = memory_stats.MemoryStats()
        self.supports_memory_budget = False
        try:
            self.instance = vk.create_instance(vk.InstanceCreateInfo(
                application_info=vk.ApplicationInfo(
                    application_name=app_name, application_version=1, engine_name=app_name,
                    engine_version=1, api_version=cvk.API_VERSION_1_3),
                enabled_layer_names=[], enabled_extension_names=[]))
            self._pick_physical_device()
            features = vk.PhysicalDeviceFeatures2(
                features=_native.feature_struct(vk.PhysicalDeviceFeatures, shader_int64=True))
            f12 = _native.feature_struct(vk.PhysicalDeviceVulkan12Features,
                                         buffer_device_address=True)
            f13 = _native.feature_struct(vk.PhysicalDeviceVulkan13Features,
                                         synchronization2=True, maintenance4=True)
            self.device = vk.create_device(self.physical_device, vk.DeviceCreateInfo(
                p_next=_native.pnext_chain(features, f13, f12),
                queue_create_infos=[vk.DeviceQueueCreateInfo(
                    queue_family_index=self.queue_family_index, queue_priorities=[1.0])],
                enabled_extension_names=[]))
            self.queue = vk.get_device_queue(self.device, self.queue_family_index, 0)
            self.command_pool = vk.create_command_pool(self.device, vk.CommandPoolCreateInfo(
                flags=vk.CommandPoolCreateFlags.RESET_COMMAND_BUFFER,
                queue_family_index=self.queue_family_index))
            self.mem_props = vk.get_physical_device_memory_properties(self.physical_device)
        except Exception:
            self.destroy()
            raise

    def _pick_physical_device(self):
        best = None
        for device in vk.enumerate_physical_devices(self.instance):
            props = vk.get_physical_device_properties_2(device).properties
            if props.api_version < cvk.API_VERSION_1_3:
                continue
            family = next((i for i, qf in enumerate(
                vk.get_physical_device_queue_family_properties(device))
                if qf.queue_flags & vk.QueueFlags.COMPUTE), None)
            if family is None:
                continue
            discrete = props.device_type == vk.PhysicalDeviceType.DISCRETE_GPU
            if best is None or (discrete and not best[2]):
                best = (device, family, discrete, props.device_name)
        if best is None:
            raise RuntimeError('no Vulkan 1.3 device with a compute queue found')
        self.physical_device, self.queue_family_index = best[0], best[1]
        name = best[3]
        self.device_name = (name.decode(errors='replace') if isinstance(name, bytes)
                            else str(name)).rstrip('\0')

    def find_memory_type(self, type_bits, properties):
        for i in range(len(self.mem_props.memory_types)):
            if type_bits & (1 << i) and (
                    self.mem_props.memory_types[i].property_flags & properties) == properties:
                return i
        raise RuntimeError('no suitable Vulkan memory type')

    def set_debug_name(self, handle, object_type, name):
        pass

    def on_destroy(self, callback):
        self._on_destroy.append(callback)

    def _one_shot_commands(self):
        cmd = vk.allocate_command_buffers(self.device, vk.CommandBufferAllocateInfo(
            command_pool=self.command_pool, level=vk.CommandBufferLevel.PRIMARY,
            command_buffer_count=1))[0]
        vk.begin_command_buffer(cmd, vk.CommandBufferBeginInfo(
            flags=vk.CommandBufferUsageFlags.ONE_TIME_SUBMIT))
        return cmd

    def _submit_and_wait(self, cmd):
        try:
            vk.end_command_buffer(cmd)
            submit_and_wait(self, cmd)
        finally:
            vk.free_command_buffers(self.device, self.command_pool, [cmd])

    def destroy(self):
        if self.device:
            vk.device_wait_idle(self.device)
        for callback in reversed(self._on_destroy):
            callback()
        self._on_destroy.clear()
        for arena in reversed(self._arenas):
            arena.destroy()
        self._arenas.clear()
        if self.command_pool:
            vk.destroy_command_pool(self.device, self.command_pool)
            self.command_pool = None
        if self.device:
            vk.destroy_device(self.device)
            self.device = None
        if self.instance:
            vk.destroy_instance(self.instance)
            self.instance = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


def submit_and_wait(ctx, cmd, fence=None):
    """Submit one command buffer and block until it finishes."""
    own = fence is None
    if own:
        fence = vk.create_fence(ctx.device, vk.FenceCreateInfo())
    try:
        vk.queue_submit2(ctx.queue, [vk.SubmitInfo2(command_buffer_infos=[
            vk.CommandBufferSubmitInfo(command_buffer=cmd, device_mask=0)])], fence)
        vk.wait_for_fences(ctx.device, [fence], wait_all=True, timeout=0xFFFFFFFFFFFFFFFF)
    finally:
        if own:
            vk.destroy_fence(ctx.device, fence)
        else:
            vk.reset_fences(ctx.device, [fence])


class Recording:
    """A command buffer recorded once through a vabs ``Frame``, submitted repeatedly.

    ``with rec.record() as frame:`` records; ``rec.submit()`` runs and waits.
    ``rec.start()`` and ``rec.wait()`` do the two halves separately, so the host
    can queue the next recording while this one runs.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._cmd = vk.allocate_command_buffers(ctx.device, vk.CommandBufferAllocateInfo(
            command_pool=ctx.command_pool, level=vk.CommandBufferLevel.PRIMARY,
            command_buffer_count=1))[0]
        self._fence = vk.create_fence(ctx.device, vk.FenceCreateInfo())
        ctx.on_destroy(self.destroy)

    def record(self):
        rec = self

        class _Scope:
            def __enter__(self):
                vk.begin_command_buffer(rec._cmd, vk.CommandBufferBeginInfo())
                return Frame(rec._ctx, rec._cmd, None, None, None, None, 0)

            def __exit__(self, *exc):
                vk.end_command_buffer(rec._cmd)

        return _Scope()

    def submit(self):
        submit_and_wait(self._ctx, self._cmd, self._fence)

    def start(self):
        """Submit without waiting; ``wait()`` must follow before the next submission."""
        vk.queue_submit2(self._ctx.queue, [vk.SubmitInfo2(command_buffer_infos=[
            vk.CommandBufferSubmitInfo(command_buffer=self._cmd, device_mask=0)])], self._fence)

    def wait(self):
        """Block until the submission made by ``start()`` has finished."""
        vk.wait_for_fences(self._ctx.device, [self._fence], wait_all=True,
                           timeout=0xFFFFFFFFFFFFFFFF)
        vk.reset_fences(self._ctx.device, [self._fence])

    def destroy(self):
        if self._fence is not None:
            vk.destroy_fence(self._ctx.device, self._fence)
            vk.free_command_buffers(self._ctx.device, self._ctx.command_pool, [self._cmd])
            self._fence = None


def compute_barrier(frame):
    """Order every earlier compute/transfer write before later compute/transfer/host access."""
    frame.memory_barrier(
        src_stage=vk.PipelineStageFlags2.COMPUTE_SHADER | vk.PipelineStageFlags2.COPY,
        src_access=vk.AccessFlags2.SHADER_WRITE | vk.AccessFlags2.TRANSFER_WRITE,
        dst_stage=(vk.PipelineStageFlags2.COMPUTE_SHADER | vk.PipelineStageFlags2.COPY
                   | vk.PipelineStageFlags2.HOST),
        dst_access=(vk.AccessFlags2.SHADER_READ | vk.AccessFlags2.SHADER_WRITE
                    | vk.AccessFlags2.TRANSFER_READ | vk.AccessFlags2.TRANSFER_WRITE
                    | vk.AccessFlags2.HOST_READ))
