import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import Animated, { FadeInUp, FadeOutUp } from 'react-native-reanimated';
import { XCircle, AlertTriangle, CheckCircle, AlertOctagon, LucideIcon } from 'lucide-react-native'; 

export type MessageType = 'error' | 'warning' | 'success';

interface MessageProps {
  type: MessageType;
  message: string;
  onClose: () => void;
}

const TYPE_CONFIG: Record<MessageType, { icon: LucideIcon, colors: { bg: string, border: string, text: string, icon: string } }> = {
  error: {
    icon: AlertOctagon,
    colors: {
      bg: 'bg-red-100',
      border: 'border-red-500',
      text: 'text-red-700',
      icon: '#ef4444',
    },
  },
  warning: {
    icon: AlertTriangle,
    colors: {
      bg: 'bg-yellow-100',
      border: 'border-yellow-500',
      text: 'text-yellow-700',
      icon: '#f59e0b',
    },
  },
  success: {
    icon: CheckCircle,
    colors: {
      bg: 'bg-green-100',
      border: 'border-green-500',
      text: 'text-green-700',
      icon: '#10b981',
    },
  },
};

const Message: React.FC<MessageProps> = ({ type, message, onClose }) => {
  if (!message) return null;

  const config = TYPE_CONFIG[type];
  const IconComponent = config.icon;

  return (
    <View className="absolute top-7 left-0 right-0 z-50 p-4">
      
      {/* Animated Container */}
      <Animated.View 
        entering={FadeInUp.duration(500)}
        exiting={FadeOutUp.duration(300)}
        
        className={`flex-row items-center justify-between ${config.colors.bg} border-l-4 ${config.colors.border} p-4 rounded-lg shadow-md`}
      >
        <View className="flex-row items-center flex-1 pr-4">
          <IconComponent color={config.colors.icon} size={24} className="mr-3" />
          <Text className={`${config.colors.text} font-semibold text-sm text-center`}>
            {message}
          </Text>
        </View>
        <TouchableOpacity onPress={onClose} className="p-1">
          {/* Close button icon */}
          <XCircle color={config.colors.icon} size={20} /> 
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
};

export default Message;