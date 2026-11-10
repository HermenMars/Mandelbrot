
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

const int MAX_FRAMES_IN_FLIGHT = 2;

struct PushConstants {
    float center[2] = { -0.5f, 0.0f };
    float scale = 1.5f;
    float aspect = 16.0f / 9.0f;
    int maxIterations = 300;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool complete() const {
        return graphicsFamily.has_value() &&
               presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

static std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    size_t size = static_cast<size_t>(file.tellg());

    std::vector<char> buffer(size);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));

    return buffer;
}

class Application {

public:

    void run()
    {
        initWindow();
        initVulkan();

        mainLoop();

        cleanup();
    }

private:

    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;

    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;

    VkFormat swapChainImageFormat{};
    VkExtent2D swapChainExtent{};

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT>
        imageAvailableSemaphores{};

    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT>
        renderFinishedSemaphores{};

    std::array<VkFence, MAX_FRAMES_IN_FLIGHT>
        inFlightFences{};

    size_t currentFrame = 0;

    bool framebufferResized = false;

    PushConstants camera;

    double lastTime = 0.0;

    // --------------------------------------------------
    // Window
    // --------------------------------------------------

    static void framebufferCallback(
        GLFWwindow* window,
        int width,
        int height)
    {
        auto app = reinterpret_cast<Application*>(
            glfwGetWindowUserPointer(window)
        );

        if (app)
            app->framebufferResized = true;

        (void)width;
        (void)height;
    }

    static void scrollCallback(
        GLFWwindow* window,
        double xoffset,
        double yoffset)
    {
        auto app = reinterpret_cast<Application*>(
            glfwGetWindowUserPointer(window)
        );

        if (!app)
            return;

        app->camera.scale *= static_cast<float>(
            std::pow(0.85, yoffset)
        );

        app->camera.scale = std::clamp(
            app->camera.scale,
            0.0000001f,
            10.0f
        );

        (void)xoffset;
    }

    void initWindow()
    {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(
            WIDTH,
            HEIGHT,
            "Mandelbrot Explorer",
            nullptr,
            nullptr
        );

        if (!window)
            throw std::runtime_error("Window creation failed");

        glfwSetWindowUserPointer(window, this);

        glfwSetFramebufferSizeCallback(
            window,
            framebufferCallback
        );

        glfwSetScrollCallback(
            window,
            scrollCallback
        );
    }

    // --------------------------------------------------
    // Vulkan initialization
    // --------------------------------------------------

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();

        createSwapChain();
        createImageViews();

        createRenderPass();
        createGraphicsPipeline();

        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance()
    {
        VkApplicationInfo appInfo{};

        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

        appInfo.pApplicationName = "Mandelbrot Explorer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);

        appInfo.pEngineName = "Custom Vulkan Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);

        appInfo.apiVersion = VK_API_VERSION_1_0;

        uint32_t extensionCount = 0;

        const char** extensions =
            glfwGetRequiredInstanceExtensions(&extensionCount);

        VkInstanceCreateInfo createInfo{};

        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        createInfo.pApplicationInfo = &appInfo;

        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;

        createInfo.enabledLayerCount = 0;

        if (vkCreateInstance(
            &createInfo,
            nullptr,
            &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan instance");
        }
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(
            instance,
            window,
            nullptr,
            &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    // --------------------------------------------------
    // Physical device selection
    // --------------------------------------------------

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice gpu)
    {
        QueueFamilyIndices indices;

        uint32_t count = 0;

        vkGetPhysicalDeviceQueueFamilyProperties(
            gpu,
            &count,
            nullptr
        );

        std::vector<VkQueueFamilyProperties> families(count);

        vkGetPhysicalDeviceQueueFamilyProperties(
            gpu,
            &count,
            families.data()
        );

        for (uint32_t i = 0; i < count; i++)
        {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.graphicsFamily = i;

            VkBool32 presentSupport = false;

            vkGetPhysicalDeviceSurfaceSupportKHR(
                gpu,
                i,
                surface,
                &presentSupport
            );

            if (presentSupport)
                indices.presentFamily = i;

            if (indices.complete())
                break;
        }

        return indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice gpu)
    {
        uint32_t count = 0;

        vkEnumerateDeviceExtensionProperties(
            gpu,
            nullptr,
            &count,
            nullptr
        );

        std::vector<VkExtensionProperties> extensions(count);

        vkEnumerateDeviceExtensionProperties(
            gpu,
            nullptr,
            &count,
            extensions.data()
        );

        for (const auto& extension : extensions)
        {
            if (std::string(extension.extensionName) ==
                VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                return true;
        }

        return false;
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice gpu)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            gpu,
            surface,
            &details.capabilities
        );

        uint32_t count = 0;

        vkGetPhysicalDeviceSurfaceFormatsKHR(
            gpu,
            surface,
            &count,
            nullptr
        );

        details.formats.resize(count);

        vkGetPhysicalDeviceSurfaceFormatsKHR(
            gpu,
            surface,
            &count,
            details.formats.data()
        );

        vkGetPhysicalDeviceSurfacePresentModesKHR(
            gpu,
            surface,
            &count,
            nullptr
        );

        details.presentModes.resize(count);

        vkGetPhysicalDeviceSurfacePresentModesKHR(
            gpu,
            surface,
            &count,
            details.presentModes.data()
        );

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice gpu)
    {
        QueueFamilyIndices indices = findQueueFamilies(gpu);

        if (!indices.complete())
            return false;

        if (!checkDeviceExtensionSupport(gpu))
            return false;

        auto details = querySwapChainSupport(gpu);

        return !details.formats.empty() &&
               !details.presentModes.empty();
    }

    void pickPhysicalDevice()
    {
        uint32_t count = 0;

        vkEnumeratePhysicalDevices(
            instance,
            &count,
            nullptr
        );

        if (count == 0)
            throw std::runtime_error("No Vulkan GPU found");

        std::vector<VkPhysicalDevice> devices(count);

        vkEnumeratePhysicalDevices(
            instance,
            &count,
            devices.data()
        );

        for (auto gpu : devices)
        {
            if (isDeviceSuitable(gpu))
            {
                physicalDevice = gpu;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("No suitable Vulkan GPU found");

        VkPhysicalDeviceProperties properties{};

        vkGetPhysicalDeviceProperties(
            physicalDevice,
            &properties
        );

        std::cout << "Using GPU: "
                  << properties.deviceName << '\n';
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices =
            findQueueFamilies(physicalDevice);

        std::set<uint32_t> uniqueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        float priority = 1.0f;

        std::vector<VkDeviceQueueCreateInfo> queueInfos;

        for (uint32_t family : uniqueFamilies)
        {
            VkDeviceQueueCreateInfo queueInfo{};

            queueInfo.sType =
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;

            queueInfo.queueFamilyIndex = family;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;

            queueInfos.push_back(queueInfo);
        }

        VkPhysicalDeviceFeatures features{};

        const char* extensions[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        VkDeviceCreateInfo createInfo{};

        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount =
            static_cast<uint32_t>(queueInfos.size());

        createInfo.pQueueCreateInfos = queueInfos.data();

        createInfo.pEnabledFeatures = &features;

        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = extensions;

        if (vkCreateDevice(
            physicalDevice,
            &createInfo,
            nullptr,
            &device) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(
            device,
            indices.graphicsFamily.value(),
            0,
            &graphicsQueue
        );

        vkGetDeviceQueue(
            device,
            indices.presentFamily.value(),
            0,
            &presentQueue
        );
    }

    // --------------------------------------------------
    // Swapchain
    // --------------------------------------------------

    VkSurfaceFormatKHR chooseSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& formats)
    {
        for (const auto& format : formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
        }

        return formats[0];
    }

    VkPresentModeKHR choosePresentMode(
        const std::vector<VkPresentModeKHR>& modes)
    {
        for (auto mode : modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(
        const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
            return capabilities.currentExtent;

        int width = 0;
        int height = 0;

        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D extent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        extent.width = std::clamp(
            extent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width
        );

        extent.height = std::clamp(
            extent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height
        );

        return extent;
    }

    void createSwapChain()
    {
        auto details = querySwapChainSupport(physicalDevice);

        auto format = chooseSurfaceFormat(details.formats);

        auto presentMode = choosePresentMode(details.presentModes);

        auto extent = chooseExtent(details.capabilities);

        uint32_t imageCount =
            details.capabilities.minImageCount + 1;

        if (details.capabilities.maxImageCount > 0)
        {
            imageCount = std::min(
                imageCount,
                details.capabilities.maxImageCount
            );
        }

        QueueFamilyIndices indices =
            findQueueFamilies(physicalDevice);

        uint32_t families[] = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        VkSwapchainCreateInfoKHR info{};

        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;

        info.surface = surface;

        info.minImageCount = imageCount;

        info.imageFormat = format.format;
        info.imageColorSpace = format.colorSpace;

        info.imageExtent = extent;
        info.imageArrayLayers = 1;

        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        if (indices.graphicsFamily != indices.presentFamily)
        {
            info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;

            info.queueFamilyIndexCount = 2;
            info.pQueueFamilyIndices = families;
        }
        else
        {
            info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        info.preTransform = details.capabilities.currentTransform;

        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        info.presentMode = presentMode;

        info.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(
            device,
            &info,
            nullptr,
            &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swapchain");
        }

        vkGetSwapchainImagesKHR(
            device,
            swapChain,
            &imageCount,
            nullptr
        );

        swapChainImages.resize(imageCount);

        vkGetSwapchainImagesKHR(
            device,
            swapChain,
            &imageCount,
            swapChainImages.data()
        );

        swapChainImageFormat = format.format;
        swapChainExtent = extent;

        camera.aspect =
            static_cast<float>(extent.width) /
            static_cast<float>(extent.height);
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo info{};

            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

            info.image = swapChainImages[i];

            info.viewType = VK_IMAGE_VIEW_TYPE_2D;

            info.format = swapChainImageFormat;

            info.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY
            };

            info.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;

            info.subresourceRange.baseMipLevel = 0;
            info.subresourceRange.levelCount = 1;

            info.subresourceRange.baseArrayLayer = 0;
            info.subresourceRange.layerCount = 1;

            if (vkCreateImageView(
                device,
                &info,
                nullptr,
                &swapChainImageViews[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image view");
            }
        }
    }

    // --------------------------------------------------
    // Render pass
    // --------------------------------------------------

    void createRenderPass()
    {
        VkAttachmentDescription attachment{};

        attachment.format = swapChainImageFormat;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;

        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        attachment.finalLayout =
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorReference{};

        colorReference.attachment = 0;

        colorReference.layout =
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};

        subpass.pipelineBindPoint =
            VK_PIPELINE_BIND_POINT_GRAPHICS;

        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;

        VkSubpassDependency dependency{};

        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        dependency.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo info{};

        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

        info.attachmentCount = 1;
        info.pAttachments = &attachment;

        info.subpassCount = 1;
        info.pSubpasses = &subpass;

        info.dependencyCount = 1;
        info.pDependencies = &dependency;

        if (vkCreateRenderPass(
            device,
            &info,
            nullptr,
            &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create render pass");
        }
    }

    // --------------------------------------------------
    // Graphics pipeline
    // --------------------------------------------------

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo info{};

        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

        info.codeSize = code.size();

        info.pCode = reinterpret_cast<const uint32_t*>(
            code.data()
        );

        VkShaderModule shader = VK_NULL_HANDLE;

        if (vkCreateShaderModule(
            device,
            &info,
            nullptr,
            &shader) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create shader module");
        }

        return shader;
    }

    void createGraphicsPipeline()
    {
        auto vertCode = readFile(
            std::string(SHADER_DIR) + "/mandelbrot.vert.spv"
        );

        auto fragCode = readFile(
            std::string(SHADER_DIR) + "/mandelbrot.frag.spv"
        );

        VkShaderModule vertShader = createShaderModule(vertCode);
        VkShaderModule fragShader = createShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo vertStage{};

        vertStage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

        vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertShader;
        vertStage.pName = "main";

        VkPipelineShaderStageCreateInfo fragStage{};

        fragStage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragShader;
        fragStage.pName = "main";

        VkPipelineShaderStageCreateInfo stages[] = {
            vertStage,
            fragStage
        };

        VkPipelineVertexInputStateCreateInfo vertexInput{};

        vertexInput.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo assembly{};

        assembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};

        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);

        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};

        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};

        viewportState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;

        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};

        rasterizer.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

        rasterizer.lineWidth = 1.0f;

        rasterizer.cullMode = VK_CULL_MODE_NONE;

        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};

        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlend{};

        colorBlend.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        colorBlend.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};

        colorBlending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlend;

        VkPushConstantRange pushRange{};

        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};

        layoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(
            device,
            &layoutInfo,
            nullptr,
            &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};

        pipelineInfo.sType =
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stages;

        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;

        pipelineInfo.pColorBlendState = &colorBlending;

        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        VkResult result = vkCreateGraphicsPipelines(
            device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &graphicsPipeline
        );

        vkDestroyShaderModule(device, vertShader, nullptr);
        vkDestroyShaderModule(device, fragShader, nullptr);

        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");
    }

    // --------------------------------------------------
    // Framebuffers
    // --------------------------------------------------

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo info{};

            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;

            info.renderPass = renderPass;

            info.attachmentCount = 1;
            info.pAttachments = attachments;

            info.width = swapChainExtent.width;
            info.height = swapChainExtent.height;

            info.layers = 1;

            if (vkCreateFramebuffer(
                device,
                &info,
                nullptr,
                &swapChainFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create framebuffer");
            }
        }
    }

    // --------------------------------------------------
    // Command buffers
    // --------------------------------------------------

    void createCommandPool()
    {
        QueueFamilyIndices indices =
            findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo info{};

        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        info.queueFamilyIndex = indices.graphicsFamily.value();

        if (vkCreateCommandPool(
            device,
            &info,
            nullptr,
            &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool");
        }
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo info{};

        info.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

        info.commandPool = commandPool;

        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

        if (vkAllocateCommandBuffers(
            device,
            &info,
            commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffers");
        }
    }

    void recordCommandBuffer(
        VkCommandBuffer commandBuffer,
        uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo begin{};

        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(
            commandBuffer,
            &begin) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin command buffer");
        }

        VkClearValue clear{};

        clear.color = {{ 0.0f, 0.0f, 0.0f, 1.0f }};

        VkRenderPassBeginInfo renderBegin{};

        renderBegin.sType =
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

        renderBegin.renderPass = renderPass;

        renderBegin.framebuffer =
            swapChainFramebuffers[imageIndex];

        renderBegin.renderArea.extent = swapChainExtent;

        renderBegin.clearValueCount = 1;
        renderBegin.pClearValues = &clear;

        vkCmdBeginRenderPass(
            commandBuffer,
            &renderBegin,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            graphicsPipeline
        );

        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(PushConstants),
            &camera
        );

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer");
    }

    // --------------------------------------------------
    // Synchronization
    // --------------------------------------------------

    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreInfo{};

        semaphoreInfo.sType =
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};

        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(
                device,
                &semaphoreInfo,
                nullptr,
                &imageAvailableSemaphores[i]) != VK_SUCCESS ||

                vkCreateSemaphore(
                device,
                &semaphoreInfo,
                nullptr,
                &renderFinishedSemaphores[i]) != VK_SUCCESS ||

                vkCreateFence(
                device,
                &fenceInfo,
                nullptr,
                &inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create sync objects");
            }
        }
    }

    // --------------------------------------------------
    // Camera controls
    // --------------------------------------------------

    void updateCamera(float deltaTime)
    {
        float movement = camera.scale * 1.5f * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.center[1] += movement;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.center[1] -= movement;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.center[0] -= movement * camera.aspect;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.center[0] += movement * camera.aspect;

        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS)
            camera.scale *= 0.98f;

        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS)
            camera.scale *= 1.02f;

        camera.scale = std::clamp(
            camera.scale,
            0.0000001f,
            10.0f
        );

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            camera.maxIterations = std::min(
                camera.maxIterations + 1,
                10000
            );
        }

        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            camera.maxIterations = std::max(
                camera.maxIterations - 1,
                10
            );
        }

        static bool resetPressed = false;

        bool resetDown =
            glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;

        if (resetDown && !resetPressed)
        {
            camera.center[0] = -0.5f;
            camera.center[1] = 0.0f;

            camera.scale = 1.5f;
            camera.maxIterations = 300;
        }

        resetPressed = resetDown;
    }

    // --------------------------------------------------
    // Rendering
    // --------------------------------------------------

    void drawFrame()
    {
        vkWaitForFences(
            device,
            1,
            &inFlightFences[currentFrame],
            VK_TRUE,
            UINT64_MAX
        );

        uint32_t imageIndex;

        VkResult result = vkAcquireNextImageKHR(
            device,
            swapChain,
            UINT64_MAX,
            imageAvailableSemaphores[currentFrame],
            VK_NULL_HANDLE,
            &imageIndex
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }

        if (result != VK_SUCCESS &&
            result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("Failed to acquire swapchain image");
        }

        vkResetFences(
            device,
            1,
            &inFlightFences[currentFrame]
        );

        vkResetCommandBuffer(
            commandBuffers[currentFrame],
            0
        );

        recordCommandBuffer(
            commandBuffers[currentFrame],
            imageIndex
        );

        VkSemaphore waitSemaphores[] = {
            imageAvailableSemaphores[currentFrame]
        };

        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };

        VkSemaphore signalSemaphores[] = {
            renderFinishedSemaphores[currentFrame]
        };

        VkSubmitInfo submit{};

        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = waitSemaphores;

        submit.pWaitDstStageMask = waitStages;

        submit.commandBufferCount = 1;

        submit.pCommandBuffers =
            &commandBuffers[currentFrame];

        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(
            graphicsQueue,
            1,
            &submit,
            inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit draw command");
        }

        VkPresentInfoKHR present{};

        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = signalSemaphores;

        present.swapchainCount = 1;
        present.pSwapchains = &swapChain;

        present.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &present);

        if (result == VK_ERROR_OUT_OF_DATE_KHR ||
            result == VK_SUBOPTIMAL_KHR ||
            framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to present image");
        }

        currentFrame =
            (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // --------------------------------------------------
    // Swapchain recreation
    // --------------------------------------------------

    void cleanupSwapChain()
    {
        for (auto framebuffer : swapChainFramebuffers)
            vkDestroyFramebuffer(device, framebuffer, nullptr);

        for (auto imageView : swapChainImageViews)
            vkDestroyImageView(device, imageView, nullptr);

        if (graphicsPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, graphicsPipeline, nullptr);

        if (pipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        if (renderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(device, renderPass, nullptr);

        if (swapChain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device, swapChain, nullptr);

        swapChainFramebuffers.clear();
        swapChainImageViews.clear();
        swapChainImages.clear();
    }

    void recreateSwapChain()
    {
        int width = 0;
        int height = 0;

        glfwGetFramebufferSize(window, &width, &height);

        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
    }

    // --------------------------------------------------
    // Main loop
    // --------------------------------------------------

    void mainLoop()
    {
        lastTime = glfwGetTime();

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            double currentTime = glfwGetTime();

            float deltaTime =
                static_cast<float>(currentTime - lastTime);

            lastTime = currentTime;

            updateCamera(deltaTime);

            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    // --------------------------------------------------
    // Cleanup
    // --------------------------------------------------

    void cleanup()
    {
        if (device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);

            cleanupSwapChain();

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                vkDestroySemaphore(
                    device,
                    imageAvailableSemaphores[i],
                    nullptr
                );

                vkDestroySemaphore(
                    device,
                    renderFinishedSemaphores[i],
                    nullptr
                );

                vkDestroyFence(
                    device,
                    inFlightFences[i],
                    nullptr
                );
            }

            vkDestroyCommandPool(device, commandPool, nullptr);

            vkDestroyDevice(device, nullptr);
        }

        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);

        if (instance != VK_NULL_HANDLE)
            vkDestroyInstance(instance, nullptr);

        if (window)
            glfwDestroyWindow(window);

        glfwTerminate();
    }
};

// --------------------------------------------------
// Entry point
// --------------------------------------------------

int main()
{
    try
    {
        Application app;
        app.run();
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: "
                  << exception.what() << '\n';

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
