import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { TwitterModule } from './twitter/twitter.module';

@Module({
  imports: [TwitterModule],
  controllers: [AppController],
})
export class AppModule {}
