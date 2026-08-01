// clang-format off
#include <sn/core.hpp>
#include <sn/io.hpp>

#include "vendor/imgui/imgui_impl_glfw.h"
#include "vendor/imgui/imgui_impl_vulkan.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <fstream>
#include <set>
#include <optional>
#include <limits>
#include <stdexcept>
#include <array>
#include <random>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstring>

using namespace sn::types;

// cfg
static constexpr u32 kInitialWidth = 1920;
static constexpr u32 kInitialHeight = 1080;
static constexpr i32 kFramesInFlight = 2;

static constexpr i32 kGridHalfExtent = 200;
static constexpr f32 kGridSpacing = 1.0f;
static constexpr f32 kFogDensity = 0.010f;

#ifdef NDEBUG
static constexpr bool kEnableValidation = false;
#else
static constexpr bool kEnableValidation = true;
#endif

static const std::vector<const char*> kValidationLayers = 
{
  "VK_LAYER_KHRONOS_validation"
};
static const std::vector<const char*> kDeviceExtensions = 
{
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME
};

#define VK_CHECK(call)                                                                                         \
  do {                                                                                                         \
    VkResult _res = (call);                                                                                    \
    if (_res != VK_SUCCESS) {                                                                                  \
      throw std::runtime_error(std::string("Vulkan call failed (") + std::to_string(int(_res)) + "): " #call); \
    }                                                                                                          \
  } while (0)                                                                                                  \

// ----------------------------------
// Data
// ----------------------------------

struct Vertex
{
  glm::vec3 pos;
  glm::vec3 color;

  static VkVertexInputBindingDescription bindingDescription()
  {
    VkVertexInputBindingDescription b{};
    b.binding = 0;
    b.stride = sizeof(Vertex);
    b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return b;
  }

  static std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions()
  {
    std::array<VkVertexInputAttributeDescription, 2> a{};
    a[0].location = 0;
    a[0].binding = 0;
    a[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    a[0].offset = offsetof(Vertex, pos);
    a[1].location = 1;
    a[1].binding = 0;
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    a[1].offset = offsetof(Vertex, color);
    return a;
  }
};

struct PushConstants
{
  glm::mat4 viewProj;
  glm::vec4 camPosFog; // xyz = camera world position, w = fog density
};

struct Camera
{
  glm::vec3 position{0.0f, 6.0f, 24.0f};
  f32 yaw = -90.f;
  f32 pitch = -8.0f;
  f32 fovY = 90.f;

  glm::vec3 forward() const
  {
    f32 cy = std::cos(glm::radians(yaw));
    f32 sy = std::sin(glm::radians(yaw));
    f32 cp = std::cos(glm::radians(pitch));
    f32 sp = std::sin(glm::radians(pitch));
    return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
  }
  glm::vec3 right() const
  {
    return glm::normalize(glm::cross(forward(), glm::vec3(0, 1, 0)));
  }
  glm::mat4 view() const
  {
    return glm::lookAt(position, position + forward(), glm::vec3(0, 1, 0));
  }
};

// ----------------------------------
// Helpers
// ----------------------------------

static std::vector<char> readFile(const std::string& path)
{
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("failed to open file: " + path);
  size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback
(
 VkDebugUtilsMessageSeverityFlagBitsEXT severity,
 VkDebugUtilsMessageTypeFlagsEXT,
 const VkDebugUtilsMessengerCallbackDataEXT* data,
 void*
)
{
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    sn::err("[vulkan] {}", data->pMessage);
  return VK_FALSE;
}

class Application
{
  public:
    void run()
    {
      initWindow();
      initVulkan();
      mainLoop();
      cleanup();
    }

  private:
    // --- window / input ---
    GLFWwindow* window = nullptr;
    bool frameBufferResized = false;
    bool firstMouse = true;
    double lastMouseX = 0.0, lastMouseY = 0.0;
    bool f1Held = false;
    bool mouseCaptured = true;
    Camera camera;

    // --- core objects ---
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice pDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    u32 graphicsFamily = 0, presentFamily = 0;
    VkQueue graphicsQueue = VK_NULL_HANDLE, presentQueue = VK_NULL_HANDLE;

    // --- swapchain ---
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    // --- depth ---
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;

    // --- pipeline ---
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // --- geometry ---
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    u32 vertexCount = 0;

    // --- commands & sync ---
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailable; // per image in flight
    std::vector<VkSemaphore> renderFinished; // per swapchain image
    std::vector<VkFence> inFlight;           // per frame in flight
    u32 currentFrame = 0;

    // ----------------------------------
    // Window
    // ----------------------------------
    void initWindow()
    {
      if (!glfwInit())
        throw std::runtime_error("failed to initialize glfw");
      if (!glfwVulkanSupported())
        throw std::runtime_error("no vulkan loader found");

      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      window = glfwCreateWindow(kInitialWidth, kInitialHeight, "Window", nullptr, nullptr);

      if (!window)
        throw std::runtime_error("failed to create window");

      glfwSetWindowUserPointer(window, this);
      glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int)
          {
            static_cast<Application*>(glfwGetWindowUserPointer(w))->frameBufferResized = true;
          });

      glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
      if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    // ----------------------------------
    // ImGui
    // ----------------------------------
    void initImGui()
    {
      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGui::StyleColorsDark();
      ImGui_ImplGlfw_InitForVulkan(window, true);

      ImGui_ImplVulkan_InitInfo info{};
      info.ApiVersion = VK_API_VERSION_1_3;
      info.Instance = instance;
      info.PhysicalDevice = pDevice;
      info.Device = device;
      info.QueueFamily = graphicsFamily;
      info.Queue = graphicsQueue;
      info.DescriptorPoolSize = 16;
      info.MinImageCount = static_cast<u32>(swapchainImages.size());
      info.ImageCount = static_cast<u32>(swapchainImages.size());
      info.UseDynamicRendering = true;

      auto& pInfo = info.PipelineInfoMain;
      pInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
      pInfo.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
      pInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
      pInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;
      pInfo.PipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat;

      ImGui_ImplVulkan_Init(&info);
    }

    // ----------------------------------
    // Vulkan
    // ----------------------------------
    void initVulkan()
    {
      createInstance();
      setupDebugMessenger();
      createSurface();
      pickPhysicalDevice();
      createLogicalDevice();
      createSwapchain();
      createImageViews();
      createDepthResources();
      createPipeline();
      createGeometry();
      createCommandPool();
      createCommandBuffers();
      createSyncObjects();
      initImGui();
    }

    void createInstance()
    {
      u32 apiVersion = VK_API_VERSION_1_0;
      vkEnumerateInstanceVersion(&apiVersion);
      if (apiVersion < VK_API_VERSION_1_3)
        throw std::runtime_error("Vulkan 1.3 support required");

      VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
      appInfo.pApplicationName = "vulkan stuff";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_3;

      u32 glfwExtCount = 0;
      const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
      std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

      bool validation = kEnableValidation && validationLayersAvailable();
      if (validation)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

      VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      ci.pApplicationInfo = &appInfo;
      ci.enabledExtensionCount = static_cast<u32>(extensions.size());
      ci.ppEnabledExtensionNames = extensions.data();

      VkDebugUtilsMessengerCreateInfoEXT dbg = debugMessengerCreateInfo();
      if (validation)
      {
        ci.enabledLayerCount = static_cast<u32>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
        ci.pNext = &dbg;
      }

      VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
    }

    static bool validationLayersAvailable()
    {
      u32 count = 0;
      vkEnumerateInstanceLayerProperties(&count, nullptr);
      std::vector<VkLayerProperties> available(count);
      vkEnumerateInstanceLayerProperties(&count, available.data());
      for (const char* wanted : kValidationLayers)
      {
        bool found = false;
        for (const auto& props : available)
          if (std::strcmp(wanted, props.layerName) == 0) { found = true; break; }
        if (!found)
        {
          sn::err("[info] validation layers not installed. continuing without\n");
          return false;
        }
      }
      return true;
    }

    static VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo()
    {
      VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      ci.pfnUserCallback = debugCallback;
      return ci;
    }

    void setupDebugMessenger()
    {
      if (!kEnableValidation) return;
      auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
      if (!fn) return;
      VkDebugUtilsMessengerCreateInfoEXT ci = debugMessengerCreateInfo();
      fn(instance, &ci, nullptr, &debugMessenger);
    }

    void createSurface()
    {
      VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface)); 
    }

    struct QueueIndices
    {
      std::optional<u32> graphics;
      std::optional<u32> present;
      bool complete() const { return graphics.has_value() && present.has_value(); }
    };

    QueueIndices findQueueFamilies(VkPhysicalDevice dev) const
    {
      QueueIndices idx;
      u32 count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
      std::vector<VkQueueFamilyProperties> families(count);
      vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

      for (u32 i = 0; i < count; ++i)
      {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
          idx.graphics = i;
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present);
        if (present) idx.present = i;

        if (idx.graphics.has_value() && idx.graphics == idx.present)
        {
          idx.present = i;
          break;
        }
      }
      return idx;
    }

    bool deviceSuitable(VkPhysicalDevice dev) const
    {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(dev, &props);
      if (props.apiVersion < VK_API_VERSION_1_3) return false;
      if (!findQueueFamilies(dev).complete()) return false;

      // swapchain extension present?
      u32 extCount = 0;
      vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
      std::vector<VkExtensionProperties> exts(extCount);
      vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
      std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
      for (const auto& e : exts) required.erase(e.extensionName);
      if (!required.empty()) return false;

      // surface actually usable?
      u32 fmtCount = 0, modeCount = 0;
      vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &fmtCount, nullptr);
      vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &modeCount, nullptr);
      if (fmtCount == 0 || modeCount == 0) return false;

      // check for the 1.3 features ig
      VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &f13;
      vkGetPhysicalDeviceFeatures2(dev, &f2);
      return f13.dynamicRendering && f13.synchronization2;
    }

    void pickPhysicalDevice()
    {
      u32 count = 0;
      vkEnumeratePhysicalDevices(instance, &count, nullptr);
      if (count == 0) throw std::runtime_error("no vulkan-capable GPU found");
      std::vector<VkPhysicalDevice> devices(count);
      vkEnumeratePhysicalDevices(instance, &count, devices.data());

      for (auto dev : devices)
      {
        if (!deviceSuitable(dev)) continue;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        if (pDevice == VK_NULL_HANDLE || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
          pDevice = dev;
          if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
        }
      }
      if (pDevice == VK_NULL_HANDLE)
        throw std::runtime_error("no GPU with Vulkan 1.3 + dynamicRendering + synchronization2");

      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(pDevice, &props);
      sn::log("GPU: {}", props.deviceName);

      auto idx = findQueueFamilies(pDevice);
      graphicsFamily = *idx.graphics;
      presentFamily = *idx.present;
    }

    void createLogicalDevice()
    {
      std::set<u32> uniqueFamilies{graphicsFamily, presentFamily};
      std::vector<VkDeviceQueueCreateInfo> queueInfos;
      f32 priority = 1.0f;
      for (u32 fam : uniqueFamilies)
      {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = fam;
        q.queueCount = 1;
        q.pQueuePriorities = &priority;
        queueInfos.push_back(q);
      }

      VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      f13.dynamicRendering = VK_TRUE;
      f13.synchronization2 = VK_TRUE;

      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &f13;

      VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
      ci.pNext = &f2;
      ci.queueCreateInfoCount = static_cast<u32>(queueInfos.size());
      ci.pQueueCreateInfos = queueInfos.data();
      ci.enabledExtensionCount = static_cast<u32>(kDeviceExtensions.size());
      ci.ppEnabledExtensionNames = kDeviceExtensions.data();

      VK_CHECK(vkCreateDevice(pDevice, &ci, nullptr, &device));
      vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
      vkGetDeviceQueue(device, presentFamily, 0, &presentQueue);
    }

    // ----------------------------------
    // Swapchain
    // ----------------------------------
    void createSwapchain()
    {
      VkSurfaceCapabilitiesKHR caps{};
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pDevice, surface, &caps);

      u32 fmtCount = 0;
      vkGetPhysicalDeviceSurfaceFormatsKHR(pDevice, surface, &fmtCount, nullptr);
      std::vector<VkSurfaceFormatKHR> formats(fmtCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(pDevice, surface, &fmtCount, formats.data());

      VkSurfaceFormatKHR chosen = formats[0];
      for (const auto& f : formats)
      {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
          chosen = f;
          break;
        }
      }

      u32 modeCount = 0;
      vkGetPhysicalDeviceSurfacePresentModesKHR(pDevice, surface, &modeCount, nullptr);
      std::vector<VkPresentModeKHR> modes(modeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(pDevice, surface, &modeCount, modes.data());
      VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR; // default; always available
      for (auto m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = m; break; }

      VkExtent2D extent = caps.currentExtent;
      if (extent.width == std::numeric_limits<u32>::max())
      {
        i32 w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        extent.width = std::clamp(u32(w), caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(u32(h), caps.minImageExtent.height, caps.maxImageExtent.height);
      }

      u32 imageCount = caps.minImageCount + 1;
      if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

      VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
      ci.surface = surface;
      ci.minImageCount = imageCount;
      ci.imageFormat = chosen.format;
      ci.imageColorSpace = chosen.colorSpace;
      ci.imageExtent = extent;
      ci.imageArrayLayers = 1;
      ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      ci.preTransform = caps.currentTransform;
      ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
      ci.presentMode = presentMode;
      ci.clipped = VK_TRUE;
      ci.oldSwapchain = VK_NULL_HANDLE;

      u32 families[] = {graphicsFamily, presentFamily};
      if (graphicsFamily != presentFamily)
      {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = families;
      }
      else
      {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      }

      VK_CHECK(vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain));

      u32 actual = 0;
      vkGetSwapchainImagesKHR(device, swapchain, &actual, nullptr);
      swapchainImages.resize(actual);
      vkGetSwapchainImagesKHR(device, swapchain, &actual, swapchainImages.data());

      swapchainFormat = chosen.format;
      swapchainExtent = extent;
    }

    void createImageViews()
    {
      swapchainImageViews.resize(swapchainImages.size());
      for (size_t i = 0; i < swapchainImages.size(); ++i)
      {
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = swapchainImages[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = swapchainFormat;
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device, &ci, nullptr, &swapchainImageViews[i]));
      }
    }

    u32 findMemoryType(u32 typeBits, VkMemoryPropertyFlags props) const
    {
      VkPhysicalDeviceMemoryProperties memProps{};
      vkGetPhysicalDeviceMemoryProperties(pDevice, &memProps);
      for (u32 i = 0; i < memProps.memoryTypeCount; ++i)
      {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
          return i;
      }
      throw std::runtime_error("no suitable memory type");
    }

    void createDepthResources()
    {
      const VkFormat candidates[] =
      {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
      };
      depthFormat = VK_FORMAT_UNDEFINED;
      for (VkFormat f : candidates)
      {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(pDevice, f, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
        {
          depthFormat = f;
          break;
        }
      }
      if (depthFormat == VK_FORMAT_UNDEFINED)
        throw std::runtime_error("no supported depth format");

      VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ici.imageType = VK_IMAGE_TYPE_2D;
      ici.format = depthFormat;
      ici.extent = {swapchainExtent.width, swapchainExtent.height, 1};
      ici.mipLevels = 1;
      ici.arrayLayers = 1;
      ici.samples = VK_SAMPLE_COUNT_1_BIT;
      ici.tiling = VK_IMAGE_TILING_OPTIMAL;
      ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      VK_CHECK(vkCreateImage(device, &ici, nullptr, &depthImage));

      VkMemoryRequirements req{};
      vkGetImageMemoryRequirements(device, depthImage, &req);
      VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      alloc.allocationSize = req.size;
      alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      VK_CHECK(vkAllocateMemory(device, &alloc, nullptr, &depthMemory));
      VK_CHECK(vkBindImageMemory(device, depthImage, depthMemory, 0));

      VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vci.image = depthImage;
      vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vci.format = depthFormat;
      vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vci.subresourceRange.levelCount = 1;
      vci.subresourceRange.layerCount = 1;
      VK_CHECK(vkCreateImageView(device, &vci, nullptr, &depthImageView));
    }
    
    // ----------------------------------
    // Dynamic Render Pipeline
    // ----------------------------------
    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
      VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      ci.codeSize = code.size();
      ci.pCode = reinterpret_cast<const u32*>(code.data());
      VkShaderModule mod = VK_NULL_HANDLE;
      VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
      return mod;
    }

    void createPipeline()
    {
      VkShaderModule vert = createShaderModule(readFile("shaders/grid.vert.spv"));
      VkShaderModule frag = createShaderModule(readFile("shaders/grid.frag.spv"));

      VkPipelineShaderStageCreateInfo stages[2]{};
      stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
      stages[0].module = vert;
      stages[0].pName = "main";
      stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      stages[1].module = frag;
      stages[1].pName = "main";

      auto binding = Vertex::bindingDescription();
      auto attribs = Vertex::attributeDescriptions();
      VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vi.vertexBindingDescriptionCount = 1;
      vi.pVertexBindingDescriptions = &binding;
      vi.vertexAttributeDescriptionCount = static_cast<u32>(attribs.size());
      vi.pVertexAttributeDescriptions = attribs.data();

      VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

      VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      vp.viewportCount = 1;
      vp.scissorCount = 1;

      VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      rs.polygonMode = VK_POLYGON_MODE_FILL;
      rs.cullMode = VK_CULL_MODE_NONE;
      rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
      rs.lineWidth = 1.0f; // >1 requires wideLines feature

      VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

      VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      ds.depthTestEnable = VK_TRUE;
      ds.depthWriteEnable = VK_TRUE;
      ds.depthCompareOp = VK_COMPARE_OP_LESS;
      ds.maxDepthBounds = 1.0f;

      VkPipelineColorBlendAttachmentState blendAttachment{};
      blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      blendAttachment.blendEnable = VK_FALSE;

      VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      cb.attachmentCount = 1;
      cb.pAttachments = &blendAttachment;

      VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dyn.dynamicStateCount = 2;
      dyn.pDynamicStates = dynamics;

      VkPushConstantRange push{};
      push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
      push.offset = 0;
      push.size = sizeof(PushConstants);

      VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      lci.pushConstantRangeCount = 1;
      lci.pPushConstantRanges = &push;
      VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &pipelineLayout));

      // replacement for render pass
      VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
      rendering.colorAttachmentCount = 1;
      rendering.pColorAttachmentFormats = &swapchainFormat;
      rendering.depthAttachmentFormat = depthFormat;

      VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pci.pNext = &rendering;
      pci.stageCount = 2;
      pci.pStages = stages;
      pci.pVertexInputState = &vi;
      pci.pInputAssemblyState = &ia;
      pci.pViewportState = &vp;
      pci.pRasterizationState = &rs;
      pci.pMultisampleState = &ms;
      pci.pDepthStencilState = &ds;
      pci.pColorBlendState = &cb;
      pci.pDynamicState = &dyn;
      pci.layout = pipelineLayout;
      pci.renderPass = VK_NULL_HANDLE;

      VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));

      vkDestroyShaderModule(device, frag, nullptr);
      vkDestroyShaderModule(device, vert, nullptr);
    }

    // ------------------------------------------------------------------------
    // Geometry: ground grid, coloured axes, a few wireframe boxes for parallax
    // ------------------------------------------------------------------------
    static void addLine(std::vector<Vertex>& v, glm::vec3 a, glm::vec3 b, glm::vec3 c)
    {
      v.push_back({a, c});
      v.push_back({b, c});
    }

    static void addWireBox(std::vector<Vertex>& v, glm::vec3 center, glm::vec3 half, glm::vec3 color)
    {
      glm::vec3 c[8];
      for (i32 i = 0; i < 8; ++i)
      {
        c[i] = center + glm::vec3(
            (i & 1) ? half.x : -half.x,
            (i & 2) ? half.y : -half.y,
            (i & 4) ? half.z : -half.z);
      }
      const i32 edges[12][2] =
      {{0,1},{2,3},{4,5},{6,7},
       {0,2},{1,3},{4,6},{5,7},
       {0,4},{1,5},{2,6},{3,7}
      };
      for (auto& e : edges) addLine(v, c[e[0]], c[e[1]], color);
    }

    std::vector<Vertex> buildScene() const
    {
      std::vector<Vertex> verts;
      const f32 extent = kGridHalfExtent * kGridSpacing;
      const glm::vec3 minor(0.16f, 0.18f, 0.22f);
      const glm::vec3 major(0.32f, 0.36f, 0.44f);

      for (i32 i = -kGridHalfExtent; i <= kGridHalfExtent; ++i)
      {
        if (i == 0) continue; // axes are drawn seperately
        f32 p = i * kGridSpacing;
        const glm::vec3& col = (i % 10 == 0) ? major : minor;
        addLine(verts, {p, 0, -extent}, {p, 0, extent}, col);
        addLine(verts, {-extent, 0, p}, {extent, 0, p}, col);
      }

      // X = RED; Y = GREEN; Z = BLUE
      addLine(verts, {-extent, 0, 0}, {extent, 0, 0}, {0.85f, 0.20f, 0.20f});
      addLine(verts, {0, 0, -extent}, {0, 0, extent}, {0.20f, 0.45f, 0.95f});
      addLine(verts, {0, -extent * 0.25f, 0}, {0, extent * 0.25f, 0}, {0.25f, 0.85f, 0.35f});

      // random scattered boxes so motion reads as motion
      std::mt19937 rng(1337);
      std::uniform_real_distribution<f32> posDist(-extent * 0.65f, extent * 0.65f);
      std::uniform_real_distribution<f32> sizeDist(1.5f, 6.0f);
      std::uniform_real_distribution<f32> hDist(0.0f, 18.0f);
      for (i32 i = 0; i < 90; ++i)
      {
        f32 s = sizeDist(rng);
        glm::vec3 half(s, s * 0.8f, s);
        glm::vec3 center(posDist(rng), hDist(rng) + half.y, posDist(rng));
        addWireBox(verts, center, half, glm::vec3{0.28f, 0.55f, 0.62f});
      }

      return verts;
    }

    void createGeometry()
    {
      std::vector<Vertex> verts = buildScene();
      vertexCount = static_cast<u32>(verts.size());
      VkDeviceSize size = sizeof(Vertex) * verts.size();

      // host-visible buffer
      VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bci.size = size;
      bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &vertexBuffer));

      VkMemoryRequirements req{};
      vkGetBufferMemoryRequirements(device, vertexBuffer, &req);
      VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      alloc.allocationSize = req.size;
      alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VK_CHECK(vkAllocateMemory(device, &alloc, nullptr, &vertexMemory));
      VK_CHECK(vkBindBufferMemory(device, vertexBuffer, vertexMemory, 0));

      void* mapped = nullptr;
      VK_CHECK(vkMapMemory(device, vertexMemory, 0, size, 0, &mapped));
      std::memcpy(mapped, verts.data(), static_cast<u32>(size));
      vkUnmapMemory(device, vertexMemory);

      sn::println("scene: {} line segments\n", vertexCount / 2);
    }

    // ----------------------------------
    // Commands & sync
    // ----------------------------------
    void createCommandBuffers()
    {
      commandBuffers.resize(kFramesInFlight);
      VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      ai.commandPool        = commandPool;
      ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      ai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
      VK_CHECK(vkAllocateCommandBuffers(device, &ai, commandBuffers.data()));
    }


    void createCommandPool()
    {
      VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      ci.queueFamilyIndex = graphicsFamily;
      VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &commandPool));
    }

    void createSyncObjects()
    {
      imageAvailable.resize(kFramesInFlight);
      inFlight.resize(kFramesInFlight);
      renderFinished.resize(swapchainImages.size());

      VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

      for (i32 i = 0; i < kFramesInFlight; ++i)
      {
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &imageAvailable[i]));
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &inFlight[i]));
      }
      for (size_t i = 0; i < renderFinished.size(); ++i)
      {
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &renderFinished[i]));
      } 
    }

    // ----------------------------------
    // Frame
    // ----------------------------------
    void recordCommandBuffer(VkCommandBuffer cmd, u32 imageIndex)
    {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

      // UNDEFINED -> attachment layouts
      VkImageMemoryBarrier2 toAttachment[2]{};
      toAttachment[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
      toAttachment[0].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
      toAttachment[0].srcAccessMask = 0;
      toAttachment[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      toAttachment[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      toAttachment[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      toAttachment[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      toAttachment[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toAttachment[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toAttachment[0].image = swapchainImages[imageIndex];
      toAttachment[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

      toAttachment[1] = toAttachment[0];
      toAttachment[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
      toAttachment[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      toAttachment[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      toAttachment[1].image = depthImage;
      toAttachment[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

      VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dep.imageMemoryBarrierCount = 2;
      dep.pImageMemoryBarriers = toAttachment;
      vkCmdPipelineBarrier2(cmd, &dep);

      VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      color.imageView = swapchainImageViews[imageIndex];
      color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color.clearValue.color = {{0.020f, 0.022f, 0.035f, 1.0f}};

      VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      depth.imageView = depthImageView;
      depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      depth.clearValue.depthStencil = {1.0f, 0};

      VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      ri.renderArea = {{0, 0}, swapchainExtent};
      ri.layerCount = 1;
      ri.colorAttachmentCount = 1;
      ri.pColorAttachments = &color;
      ri.pDepthAttachment = &depth;

      vkCmdBeginRendering(cmd, &ri);

      VkViewport viewport{};
      viewport.width = static_cast<f32>(swapchainExtent.width);
      viewport.height = static_cast<f32>(swapchainExtent.height);
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(cmd, 0, 1, &viewport);

      VkRect2D scissor{{0, 0}, swapchainExtent};
      vkCmdSetScissor(cmd, 0, 1, &scissor);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);

      f32 aspect = swapchainExtent.height == 0 ? 1.0f : f32(swapchainExtent.width) / f32(swapchainExtent.height);
      glm::mat4 proj = glm::perspective(glm::radians(camera.fovY), aspect, 0.1f, 600.f);
      proj[1][1] *= -1.0f; // vulkan clipspace -> +y = down
      
      PushConstants pc{};
      pc.viewProj = proj * camera.view();
      pc.camPosFog = glm::vec4(camera.position, kFogDensity);
      vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
          0, sizeof(PushConstants), &pc);

      vkCmdDraw(cmd, vertexCount, 1, 0, 0);
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
      vkCmdEndRendering(cmd);

      VkImageMemoryBarrier2 toPresent{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      toPresent.dstStageMask = 0;
      toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toPresent.image = swapchainImages[imageIndex];
      toPresent.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

      VkDependencyInfo dep2{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dep2.imageMemoryBarrierCount = 1;
      dep2.pImageMemoryBarriers = &toPresent;
      vkCmdPipelineBarrier2(cmd, &dep2);

      VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
      VK_CHECK(vkWaitForFences(device, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX));

      u32 imageIndex = 0;
      VkResult acquire = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);

      if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
      {
        recreateSwapchain();
        return;
      }
      if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("failed to acquire next image");

      VK_CHECK(vkResetFences(device, 1, &inFlight[currentFrame]));

      VkCommandBuffer cmd = commandBuffers[currentFrame];
      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      recordCommandBuffer(cmd, imageIndex);

      VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      waitInfo.semaphore = imageAvailable[currentFrame];
      waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

      VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      signalInfo.semaphore = renderFinished[imageIndex];
      signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

      VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      cmdInfo.commandBuffer = cmd;

      VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submit.waitSemaphoreInfoCount = 1;
      submit.pWaitSemaphoreInfos = &waitInfo;
      submit.commandBufferInfoCount = 1;
      submit.pCommandBufferInfos = &cmdInfo;
      submit.signalSemaphoreInfoCount = 1;
      submit.pSignalSemaphoreInfos = &signalInfo;

      VK_CHECK(vkQueueSubmit2(graphicsQueue, 1, &submit, inFlight[currentFrame]));

      VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      present.waitSemaphoreCount = 1;
      present.pWaitSemaphores = &renderFinished[imageIndex];
      present.swapchainCount = 1;
      present.pSwapchains = &swapchain;
      present.pImageIndices = &imageIndex;

      VkResult presented = vkQueuePresentKHR(presentQueue, &present);
      if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || frameBufferResized)
      {
        frameBufferResized = false;
        recreateSwapchain();
      }
      else if (presented != VK_SUCCESS)
        throw std::runtime_error("failed to present");

      currentFrame = (currentFrame + 1) % kFramesInFlight;
    }

    void cleanupSwapchain()
    {
      vkDestroyImageView(device, depthImageView, nullptr);
      vkDestroyImage(device, depthImage, nullptr);
      vkFreeMemory(device, depthMemory, nullptr);
      depthImageView = VK_NULL_HANDLE;
      depthImage = VK_NULL_HANDLE;
      depthMemory = VK_NULL_HANDLE;

      for (auto view : swapchainImageViews)
        vkDestroyImageView(device, view, nullptr);
      swapchainImageViews.clear();

      vkDestroySwapchainKHR(device, swapchain, nullptr);
      swapchain = VK_NULL_HANDLE;
    }

    void recreateSwapchain()
    {
      i32 w = 0, h = 0;
      glfwGetFramebufferSize(window, &w, &h);
      while (w == 0 || h == 0) // minimized state
      {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
      }
      vkDeviceWaitIdle(device);

      cleanupSwapchain();
      for (auto sem : renderFinished)
        vkDestroySemaphore(device, sem, nullptr);
      renderFinished.clear();

      createSwapchain();
      createImageViews();
      createDepthResources();

      VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      renderFinished.resize(swapchainImages.size());
      for (size_t i = 0; i < renderFinished.size(); ++i)
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &renderFinished[i]));
    }

    // ----------------------------------
    // Camera / input
    // ----------------------------------
    void updateCamera(f32 dt)
    {
      if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS && !f1Held)
      {
        f1Held = true;
        mouseCaptured = !mouseCaptured;
        glfwSetInputMode(window, GLFW_CURSOR, mouseCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        firstMouse = true;
      }
      if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_RELEASE) f1Held = false;
      if (!mouseCaptured) return;

      f64 mx = 0.0, my = 0.0;
      glfwGetCursorPos(window, &mx, &my);
      if (firstMouse)
      {
        lastMouseX = mx;
        lastMouseY = my;
        firstMouse = false;
      }
      f32 dx = static_cast<f32>(mx - lastMouseX);
      f32 dy = static_cast<f32>(my - lastMouseY);
      lastMouseX = mx;
      lastMouseY = my;

      const f32 sensitivity = 0.12f;
      camera.yaw += dx * sensitivity;
      camera.pitch -= dy * sensitivity;
      camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);

      f32 speed = 14.0f;
      if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 4.0f;
      if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) speed *= 0.25f;
      speed *= dt;

      glm::vec3 fwd = camera.forward();
      glm::vec3 rgt = camera.right();
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.position += fwd * speed;
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.position -= fwd * speed;
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.position += rgt * speed;
      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.position -= rgt * speed;
      if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.position.y += speed;
      if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.position.y -= speed;
      if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    void mainLoop()
    {
      f64 last = glfwGetTime();
      f64 fpsTimer = last;
      i32 frames = 0;

      while (!glfwWindowShouldClose(window))
      {
        glfwPollEvents();

        f64 now = glfwGetTime();
        f32 dt = static_cast<f32>(now - last);
        last = now;
        dt = std::min(dt, 0.1f);

        updateCamera(dt);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Debug (F1 to toggle)");
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::DragFloat3("cam", &camera.position.x, 0.1f);
        ImGui::SliderFloat("fov", &camera.fovY, 30.0f, 110.0f);
        ImGui::End();

        ImGui::Render();

        drawFrame();

        if (++frames, now - fpsTimer >= 1.0)
        {
          std::string title = "Render Pipeline Test | " + std::to_string(frames) + " fps";
          glfwSetWindowTitle(window, title.c_str());
          frames = 0;
          fpsTimer = now;
        }
      }
      vkDeviceWaitIdle(device);
    }

    // ----------------------------------
    // Teardown
    // ----------------------------------
    void cleanup()
    {
      if (device)
      {
        cleanupSwapchain();

        for (auto sem : renderFinished) vkDestroySemaphore(device, sem, nullptr);
        for (auto sem : imageAvailable) vkDestroySemaphore(device, sem, nullptr);
        for (auto fence : inFlight) vkDestroyFence(device, fence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexMemory, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDevice(device, nullptr);
      }
      if (debugMessenger)
      {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>
          (vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance, debugMessenger, nullptr);
      }
      if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
      if (instance) vkDestroyInstance(instance, nullptr);
      if (window) glfwDestroyWindow(window);
      glfwTerminate();
    }
};

auto sn_main(SN_MAIN_ARGS) -> sn::Exit
{
  Application app;
  try
  {
    app.run();
  }
  catch (const std::exception& e)
  {
    // we aint even gonna use e here lololol
    return sn::Exit::fail("Fatal error");
  }

  return sn::Exit::ok();
}
