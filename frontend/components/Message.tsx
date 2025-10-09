// Message.tsx

import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
// Import all necessary icons
import { XCircle, AlertTriangle, CheckCircle, AlertOctagon, LucideIcon } from 'lucide-react-native'; 

// Define the possible types
export type MessageType = 'error' | 'warning' | 'success';

// 1. Define the TypeScript Interface for the props
interface MessageProps {
  type: MessageType;        // The type of message: error, warning, or success
  message: string;        // The message content
  onClose: () => void;    // Function to close the message
}

// Configuration map for styles and icons
const TYPE_CONFIG: Record<MessageType, { icon: LucideIcon, colors: { bg: string, border: string, text: string, icon: string } }> = {
  error: {
    icon: AlertOctagon, // Used AlertOctagon for error, XCircle is for the close button
    colors: {
      bg: 'bg-red-100',
      border: 'border-red-500',
      text: 'text-red-700',
      icon: '#ef4444', // Red 500
    },
  },
  warning: {
    icon: AlertTriangle,
    colors: {
      bg: 'bg-yellow-100',
      border: 'border-yellow-500',
      text: 'text-yellow-700',
      icon: '#f59e0b', // Yellow 500
    },
  },
  success: {
    icon: CheckCircle,
    colors: {
      bg: 'bg-green-100',
      border: 'border-green-500',
      text: 'text-green-700',
      icon: '#10b981', // Green 500
    },
  },
};

// 2. Apply the interface and use the config map
const Message: React.FC<MessageProps> = ({ type, message, onClose }) => {
  if (!message) return null;

  const config = TYPE_CONFIG[type];
  const IconComponent = config.icon;

  return (
    // Absolute position at the top
    <View className="absolute top-7 left-0 right-0 z-50 p-4">
      {/* Dynamic styling based on type */}
      <View 
        className={`flex-row items-center justify-between ${config.colors.bg} border-l-4 ${config.colors.border} p-4 rounded-lg shadow-md`}
      >
        <View className="flex-row items-center flex-1 pr-4">
          <IconComponent color={config.colors.icon} size={24} className="mr-3" />
          <Text className={`${config.colors.text} font-semibold text-sm text-center`}>
            {message}
          </Text>
        </View>
        <TouchableOpacity onPress={onClose} className="p-1">
          {/* Close button icon uses the error color for visibility */}
          <XCircle color={config.colors.icon} size={20} /> 
        </TouchableOpacity>
      </View>
    </View>
  );
};

export default Message;