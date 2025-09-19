// next.config.ts
import type { NextConfig } from 'next';
import type { WebpackConfigContext } from 'next/dist/server/config-shared';

const nextConfig: NextConfig = {
  webpack: (config: any, { isServer }: WebpackConfigContext) => {
    if (!isServer) {
      config.module.rules.push({
        test: /\.worker\.ts$/,
        use: { loader: 'worker-loader' },
        exclude: /node_modules/,
      });
    }
    return config;
  },
};

export default nextConfig;
